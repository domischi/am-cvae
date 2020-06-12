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
