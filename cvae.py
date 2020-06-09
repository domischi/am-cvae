import tensorflow as tf
import tensorflow.keras as keras
import os
import numpy as np
import sklearn.model_selection
from hyperopt import STATUS_OK

## Get rid of some very verbose logging of TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## Fixes the "Could not create cudnn handle" error
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def cross_validated_run(parameters, train_images, test_images, n_splits=6, do_summary=False):

    image_size_x = train_images.shape[1]
    image_size_y = train_images.shape[2]
    #TRAIN_BUF = train_images.shape[0]
    #TEST_BUF = test_images.shape[0]

    class CVAE(keras.Model):
        def __init__(self, latent_dim):
            super(CVAE, self).__init__()
            self.latent_dim = parameters['latent_dim']
            depth = parameters['conv_net_conf']['depth']

            inf_net_layers = [
                    tf.keras.layers.Input(shape=(image_size_x, image_size_x, 1)),
            ]
            total_strides=1
            for i in range(depth):
                nfilters=parameters['conv_net_conf'][f'channels_{i}']
                stride=parameters['conv_net_conf'][f'stride_{i}']
                total_strides*=stride
                inf_net_layers.append(tf.keras.layers.Conv2D(filters=nfilters, kernel_size=3, strides=stride, activation='relu', padding='same'))
            inf_net_layers.append(tf.keras.layers.Flatten())
            inf_net_layers.append(tf.keras.layers.Dense(self.latent_dim + self.latent_dim))
            self.inference_net = tf.keras.Sequential(inf_net_layers)
            if do_summary:
                print(self.inference_net.summary())

            assert image_size_x % total_strides == 0
            assert image_size_y % total_strides == 0
            condensed_image_size_x = image_size_x // total_strides
            condensed_image_size_y = image_size_y // total_strides
            last_n_channels = parameters['conv_net_conf'][f'channels_{depth-1}']
            gen_net_layers = [
                    tf.keras.layers.Input(shape=(self.latent_dim,)),
                    tf.keras.layers.Dense(units=condensed_image_size_x * condensed_image_size_y * last_n_channels, activation=tf.nn.relu),
                    tf.keras.layers.Reshape(target_shape=(condensed_image_size_x, condensed_image_size_y, last_n_channels)),
            ]
            for i in range(depth)[:1:-1]:
                nfilters=parameters['conv_net_conf'][f'channels_{i-1}']
                stride=parameters['conv_net_conf'][f'stride_{i-1}']
                total_strides*=stride
                gen_net_layers.append(tf.keras.layers.Conv2DTranspose(filters=nfilters, kernel_size=3, strides=stride, activation='relu', padding='same'))
            gen_net_layers.append(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=parameters['conv_net_conf']['stride_0'], padding="SAME", activation='sigmoid'))
            self.generative_net = tf.keras.Sequential(gen_net_layers)
            if do_summary:
                print(self.generative_net.summary())
            if  self.inference_net.input_shape != self.generative_net.output_shape:
                print(self.inference_net.summary())
                print(self.generative_net.summary())
                print(parameters['conv_net_conf']['stride_0'])
                raise RuntimeError('Output-Shape and input shape are not equal!')

        def call(self, inputs):
            mean, logvar = model.encode(inputs)
            z = model.reparameterize(mean, logvar)
            output = model.decode(z)

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

        @tf.function
        def sample(self, eps=None):
            if eps is None:
                eps = tf.random.normal(shape=(100, self.latent_dim))
            return self.decode(eps)
        def reparameterize(self, mean, logvar):
            eps = tf.random.normal(shape=[parameters['BATCH_SIZE'], self.latent_dim])
            return eps * tf.exp(logvar * .5) + mean

    val_losses=[]
    kf = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=True)

    lr = parameters['learning_rate']
    if parameters['optimizer'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif parameters['optimizer'] == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
    elif parameters['optimizer'] == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    elif parameters['optimizer'] == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
    elif parameters['optimizer'] == 'ftrl':
        optimizer = tf.keras.optimizers.Ftrl(learning_rate=lr)
    else:
        raise RuntimeError(f"Unrecognized optimizer: parameters['optimizer']")


    for i, (train_index, test_index) in enumerate(kf.split(train_images)):
        print(f'Doing split {i+1}/{kf.get_n_splits()}')
        local_train_images = train_images[train_index]
        local_test_images = train_images[test_index]
        assert local_train_images.shape[0] % parameters['BATCH_SIZE'] == 0
        model = CVAE(parameters['latent_dim'])
        model.compile(optimizer=optimizer)
        fit_history = model.fit(local_train_images, local_train_images, verbose=0, epochs=5, batch_size=parameters['BATCH_SIZE'], validation_data=(local_test_images, local_test_images))
        val_losses.append(fit_history.history['val_loss'][-1])
    return {'loss': np.mean(val_losses), 'loss_variance': np.var(val_losses), 'status': STATUS_OK}
