import tensorflow as tf
import tensorflow.keras as keras
import os
import numpy as np
from load_data import get_mnist_data
from ray import tune

## Get rid of some very verbose logging of TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class CVAE(keras.Model):
    def __init__(self, config, train_set_image_shape):
        super(CVAE, self).__init__()
        self.latent_dim = config.get('latent_dim')
        self.BATCH_SIZE = config.get('BATCH_SIZE')
        image_size_x = train_set_image_shape[1]
        image_size_y = train_set_image_shape[2]
        depth = config.get('conv_net_depth', 2)

        inf_net_layers = [
                tf.keras.layers.Input(shape=(image_size_x, image_size_x, 1)),
        ]
        total_strides=1
        for i in range(depth):
            nfilters=int(config[f'channels_{i}'])
            stride=config[f'stride_{i}']
            total_strides*=stride
            inf_net_layers.append(tf.keras.layers.Conv2D(filters=nfilters, kernel_size=3, strides=(stride,stride), activation='relu', padding='same'))
        inf_net_layers.append(tf.keras.layers.Flatten())
        inf_net_layers.append(tf.keras.layers.Dense(self.latent_dim + self.latent_dim))
        self.inference_net = tf.keras.Sequential(inf_net_layers)
        if config.get('do_summary', False):
            print(self.inference_net.summary())

        assert image_size_x % total_strides == 0
        assert image_size_y % total_strides == 0
        condensed_image_size_x = image_size_x // total_strides
        condensed_image_size_y = image_size_y // total_strides
        last_n_channels = int(config[f'channels_{depth-1}'])
        gen_net_layers = [
                tf.keras.layers.Input(shape=(self.latent_dim,)),
                tf.keras.layers.Dense(units=condensed_image_size_x * condensed_image_size_y * last_n_channels, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(condensed_image_size_x, condensed_image_size_y, last_n_channels)),
        ]
        for i in range(depth)[:0:-1]:
            nfilters=int(config[f'channels_{i-1}'])
            stride=config[f'stride_{i}']
            total_strides*=stride
            gen_net_layers.append(tf.keras.layers.Conv2DTranspose(filters=nfilters, kernel_size=3, strides=(stride,stride), activation='relu', padding='same'))
        gen_net_layers.append(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(config['stride_0'],config['stride_0']), padding="SAME", activation='sigmoid'))
        self.generative_net = tf.keras.Sequential(gen_net_layers)
        if config.get('do_summary',False):
            print(self.generative_net.summary())
        if  self.inference_net.input_shape != self.generative_net.output_shape:
            print('\n'*5)
            print(self.inference_net.summary())
            print('\n'*5)
            print(self.generative_net.summary())
            print('\n'*5)
            print(config)
            print('\n'*5)
            raise RuntimeError('Output-Shape and input shape are not equal!')

    def call(self, inputs):
        mean, logvar = self.encode(inputs)
        z = self.reparameterize(mean, logvar)
        output = self.decode(z)

        # Dealing with the losses
        reconstruction_loss = tf.keras.losses.mse(inputs,output)# works better than cross_entropy
        reconstruction_loss = tf.reduce_sum(reconstruction_loss,axis=[-2,-1])
        self.add_loss(reconstruction_loss)
        kl_loss = 1 + logvar - tf.square(mean) - tf.exp(logvar)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        self.add_loss(kl_loss)
        return output
    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar
    def decode(self, z):
        return self.generative_net(z)
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(self.BATCH_SIZE, self.latent_dim))
        return self.decode(eps)
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=[self.BATCH_SIZE, self.latent_dim])
        return eps * tf.exp(logvar * .5) + mean

class CVAETrainable(tune.Trainable):
    def _setup(self, config):
        # IMPORTANT: See the above note.
        import tensorflow as tf

        ## Fixes the "Could not create cudnn handle" error
        from tensorflow.compat.v1 import ConfigProto
        from tensorflow.compat.v1 import InteractiveSession
        tf_config = ConfigProto()
        tf_config.gpu_options.allow_growth = True
        session = InteractiveSession(config=tf_config)

        ## Load data (to get shape of data)
        from load_data import get_mnist_data
        self.x_train, self.x_test = get_mnist_data(config['BATCH_SIZE'], config.get('BINARIZATION', False))
        self.BATCH_SIZE = config.get('BATCH_SIZE')

        ## Initialize model
        self.model = CVAE(config, self.x_train.shape)

        ## Initialize optimizer
        lr = config.get('learning_rate', 1e-3)
        optimizer = config.get('optimizer', 'adam')
        if optimizer == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer == 'adagrad':
            self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
        elif optimizer == 'rmsprop':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        elif optimizer == 'nadam':
            self.optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
        elif optimizer == 'ftrl':
            self.optimizer = tf.keras.optimizers.Ftrl(learning_rate=lr)
        else:
            raise RuntimeError(f"Unrecognized optimizer: {config.get('optimizer','adam')}")

        ## Initialize metric accumulators (to be displayed in tensorboard)
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.test_loss = tf.keras.metrics.Mean(name="test_loss")

        ## Compile the model with the optimizer
        self.model.compile(optimizer=optimizer)

    def _train(self):
        ## Not sure why they are required...
        self.train_loss.reset_states()
        self.test_loss.reset_states()

        ## Fit (only for 1 epoch, bc the multi epoch thing is handled by ray)
        fit_history = self.model.fit(self.x_train, self.x_train, verbose=0, epochs=1, batch_size=self.BATCH_SIZE, validation_data=(self.x_test,self.x_test))

        ## Log the losses
        loss = fit_history.history['loss'][0]
        val_loss = fit_history.history['val_loss'][0]
        self.train_loss(loss)
        self.test_loss(val_loss)

        ## Return what is required by ray
        return {
            "epoch": self.iteration,
            "loss": self.train_loss.result().numpy(),
            "test_loss":self.test_loss.result().numpy(),
        }

if __name__ == "__main__":
    # Define constants
    BATCH_SIZE = 100
    epochs = 2 
    BINARIZATION = False
    TEST_CANDIDATES = 10
    BASE_DIR = './ray_tune/'
    os.makedirs(BASE_DIR, exist_ok=True)
    get_mnist_data(BATCH_SIZE, False)
    conv_net_filters = ['8','16','32'] # Sadly have to store them as strings to cheat and register this in tensorboard
    search_space = {
        'BATCH_SIZE': BATCH_SIZE, 
        'optimizer': tune.choice(['adam', 'adagrad', 'rmsprop', 'nadam','ftrl']),
        'learning_rate': tune.loguniform(1e-6,1e-2),
        'latent_dim' : 2,
        'conv_net_depth': tune.randint(1,4+1),
        'stride_0' : tune.randint(1,2+1),
        'channels_0' : tune.choice(conv_net_filters),
        'stride_1' : tune.randint(1,2+1),
        'channels_1' : tune.choice(conv_net_filters),
        'stride_2' : 1,
        'channels_2' : tune.choice(conv_net_filters),
        'stride_3' : 1,
        'channels_3' : tune.choice(conv_net_filters),
        }
    tune.run(
        CVAETrainable,
        stop={"training_iteration": epochs},
        verbose=0,
        num_samples=TEST_CANDIDATES,
        local_dir=BASE_DIR,
        resources_per_trial={'gpu': 1},
        config=search_space)
