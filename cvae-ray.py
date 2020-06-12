import tensorflow as tf
import tensorflow.keras as keras
import os
import numpy as np
import sklearn.model_selection
#from hyperopt import STATUS_OK
from load_data import get_mnist_data
import sys

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
            eps = tf.random.normal(shape=(100, self.latent_dim))
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

        tf.summary.scalar('conv_net_depth', config['conv_net_depth'], step=1)
        from load_data import get_mnist_data
        self.x_train, self.x_test = get_mnist_data(100, config.get('BINARIZATION', False))
        #x_train, x_test = x_train / 255.0, x_test / 255.0

        # Add a channels dimension
        #x_train = x_train[..., tf.newaxis]
        #x_test = x_test[..., tf.newaxis]
        #self.train_ds = tf.data.Dataset.from_tensor_slices((x_train, x_train))
        #self.train_ds = self.train_ds.shuffle(10000).batch(config.get("batch", 100))

        #self.test_ds = tf.data.Dataset.from_tensor_slices((x_test, x_test)).batch(100)

        self.model = CVAE(config, self.x_train.shape)

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

        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.test_loss = tf.keras.metrics.Mean(name="test_loss")

        self.model.compile(optimizer=optimizer)

        #@tf.function
        #def train_step(images, labels):
        #    with tf.GradientTape() as tape:
        #        predictions = self.model(images)
        #        loss = self.loss_object(labels, predictions)
        #    gradients = tape.gradient(loss, self.model.trainable_variables)
        #    self.optimizer.apply_gradients(
        #        zip(gradients, self.model.trainable_variables))

        #    self.train_loss(loss)
        #    self.train_accuracy(labels, predictions)

        #@tf.function
        #def test_step(images, labels):
        #    predictions = self.model(images)
        #    t_loss = self.loss_object(labels, predictions)

        #    self.test_loss(t_loss)
        #    self.test_accuracy(labels, predictions)

        #self.tf_train_step = train_step
        #self.tf_test_step = test_step



    def _train(self):
        self.train_loss.reset_states()
        #self.train_accuracy.reset_states()
        self.test_loss.reset_states()
        #self.test_accuracy.reset_states()

        fit_history = self.model.fit(self.x_train, self.x_train, verbose=0, epochs=1, batch_size=100, validation_data=(self.x_test,self.x_test))

        #for idx, (images, labels) in enumerate(self.train_ds):
        #    if idx > MAX_TRAIN_BATCH:  # This is optional and can be removed.
        #        break
        #    self.tf_train_step(images, labels)

        #for test_images, test_labels in self.test_ds:
        #    self.tf_test_step(test_images, test_labels)

        loss = fit_history.history['loss'][0]
        val_loss = fit_history.history['val_loss'][0]

        self.train_loss(loss)
        self.test_loss(val_loss)

        tf.summary.scalar('test_loss', val_loss, step=1)

        return {
            "epoch": self.iteration,
            "loss": self.train_loss.result().numpy(),
            #"accuracy": self.train_accuracy.result().numpy() * 100,
            "test_loss":self.test_loss.result().numpy(),
            #"mean_accuracy": self.test_accuracy.result().numpy() * 100
        }

#def cross_validated_run(parameters, train_images, test_images, n_splits=6, do_summary=False):
#
#    image_size_x = train_images.shape[1]
#    image_size_y = train_images.shape[2]
#    #TRAIN_BUF = train_images.shape[0]
#    #TEST_BUF = test_images.shape[0]
#
#
#    val_losses=[]
#    kf = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=True)
#
#
#    for i, (train_index, test_index) in enumerate(kf.split(train_images)):
#        print(f'Doing split {i+1}/{kf.get_n_splits()}')
#        local_train_images = train_images[train_index]
#        local_test_images = train_images[test_index]
#        assert local_train_images.shape[0] % parameters['BATCH_SIZE'] == 0
#        model = CVAE(parameters['latent_dim'])
#        model.compile(optimizer=optimizer)
#        fit_history = model.fit(local_train_images, local_train_images, verbose=0, epochs=5, batch_size=parameters['BATCH_SIZE'], validation_data=(local_test_images, local_test_images))
#        val_losses.append(fit_history.history['val_loss'][-1])
#    return {'loss': np.mean(val_losses), 'loss_variance': np.var(val_losses), 'status': STATUS_OK}


if __name__ == "__main__":
    # Define constants
    BATCH_SIZE = 100
    epochs = 2 
    BINARIZATION = False
    TEST_CANDIDATES = 10
    #BASE_DIR = 'data/'
    #os.makedirs(BASE_DIR, exist_ok=True)
    get_mnist_data(BATCH_SIZE, False)
    conv_net_filters = ['8','16','32'] # Sadly have to cheat to register this in tensorboard
    search_space = {
        'BATCH_SIZE': BATCH_SIZE, 
        'optimizer': tune.choice(['adam', 'adagrad', 'rmsprop', 'nadam','ftrl']),
        'learning_rate': tune.loguniform(1e-6,1e-2),
        'latent_dim' : 2,
        #'conv_net_depth': tune.choice([1,2,3,4]),
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
        local_dir='./ray_tune/',
        resources_per_trial={'gpu': 1},
        config=search_space)

## Define the search space
#trials = hyperopt.Trials()
#conv_net_filters = [8,16,32]
#
## Do the fitting
#best = hyperopt.fmin(lambda c: cross_validated_run(c, *get_mnist_data(BATCH_SIZE, BINARIZATION), n_splits=2), search_space, algo=hyperopt.tpe.suggest, max_evals=HYPER_OPT_EVALS, trials=trials)
#
## Save the results
#timestamp = int( time.time() )
#with open(f'{BASE_DIR}/best-{timestamp}.json', 'w') as f:
#    json.dump(hyperopt.space_eval(search_space, best), f, indent=4,sort_keys=True)
#with open(f'{BASE_DIR}/trials-{timestamp}.pkl', 'wb') as f:
#    pickle.dump(trials, f)
