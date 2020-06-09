## The following code is heavily based on Google's CVAE tutorial (https://www.tensorflow.org/tutorials/generative/cvae)
## and on Keras' blog post about auto-encoders (https://blog.keras.io/building-autoencoders-in-keras.html)

import tensorflow as tf
import tensorflow.keras as keras
import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
from pprint import pprint
import sklearn.model_selection
import hyperopt
import sys
from tqdm import tqdm
import json
import pickle

## Get rid of some very verbose logging of TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## Fixes the "Could not create cudnn handle" error
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Define constants
BATCH_SIZE = 100
epochs = 4
save_freq = 2
BINARIZATION = False
BASE_DIR = 'data/'
HYPER_OPT_EVALS = 1

# ## Load the MNIST dataset
(train_images, y_train), (test_images, y_test) = tf.keras.datasets.mnist.load_data()

image_size_x = train_images.shape[1]
image_size_y = train_images.shape[2]

train_images = train_images.reshape(train_images.shape[0], image_size_x, image_size_y, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], image_size_x, image_size_y, 1).astype('float32')

# Assert the following (otherwise script crashes during learning due to hard coded reshapes)
assert train_images.shape[0] % BATCH_SIZE == 0

# Normalizing the images to the range of [0., 1.]
train_images /= 255.
test_images /= 255.

## Binarization
if BINARIZATION:
    train_images[train_images >= .5] = 1.
    train_images[train_images < .5] = 0.
    test_images[test_images >= .5] = 1.
    test_images[test_images < .5] = 0.


TRAIN_BUF = train_images.shape[0]
TEST_BUF = test_images.shape[0]

def generate_and_save_examples(model, epoch, test_input):
    predictions = model.sample(test_input)
    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.savefig(f'{BASE_DIR}/examples_at_epoch_{epoch:04d}.png')
    plt.show()


def plot_digit_classes(cvae,
                       x_test,
                       epoch):
    if cvae.latent_dim != 2:
        return
    #x_test, y_test = data
    filename = f"{BASE_DIR}/digit_classes_in_latent_space_{epoch:04d}.png"
    # display a 2D plot of the digit classes in the latent space
    z_mean, _ = cvae.encode(tf.convert_to_tensor(x_test))
    plt.figure(figsize=(6, 5))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test, cmap='Set1')
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

def plot_digits_over_latent(cvae,
                            epoch):
    if cvae.latent_dim != 2:
        return
    filename = f"{BASE_DIR}/digits_over_latent_{epoch:04d}.png"
    # display a 10x10 2D manifold of digits
    n = 10
    digit_size = image_size_x
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = cvae.decode(z_sample)[0,:,:,0]
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = x_decoded
    plt.figure(figsize=(5,5))
    start_range = digit_size // 2
    end_range = (n - 1) * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()

def plot_results(cvae, data, rand_vec, epoch):
    generate_and_save_examples(model, epoch, rand_vec)
    plot_digit_classes(cvae, data, epoch)
    plot_digits_over_latent(cvae, epoch)


def cross_validated_run(parameters, n_splits=6, do_summary=False):
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
            eps = tf.random.normal(shape=[BATCH_SIZE, self.latent_dim])
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
        assert local_train_images.shape[0] % BATCH_SIZE == 0
        model = CVAE(parameters['latent_dim'])
        model.compile(optimizer=optimizer)
        fit_history = model.fit(local_train_images, local_train_images, verbose=0, epochs=5, batch_size=BATCH_SIZE, validation_data=(local_test_images, local_test_images))
        val_losses.append(fit_history.history['val_loss'][-1])
    return {'loss': np.mean(val_losses), 'loss_variance': np.var(val_losses), 'status': hyperopt.STATUS_OK}

trials = hyperopt.Trials()
conv_net_filters = [8,16,32]
search_space = {
    'optimizer': hyperopt.hp.choice('optimizer',['adam', 'adagrad', 'rmsprop', 'nadam','ftrl']),
    'learning_rate': hyperopt.hp.loguniform('learning_rate',np.log(1e-6),np.log(1e-2)),
    'latent_dim' : 2,
    'conv_net_conf': hyperopt.hp.choice('conv_net_conf', [
        {
            'depth': 1,
            'stride_0' : hyperopt.hp.choice('stride1_0',[1,2]),
            'channels_0' : hyperopt.hp.choice('channels1_0',conv_net_filters),
        },
        {
            'depth': 2,
            'stride_0' : hyperopt.hp.choice('stride2_0',[1,2]),
            'channels_0' : hyperopt.hp.choice('channels2_0',conv_net_filters),
            'stride_1' : hyperopt.hp.choice('stride2_1',[1,2]),
            'channels_1' : hyperopt.hp.choice('channels2_1',conv_net_filters),
        },
        {
            'depth': 3,
            'stride_0' : hyperopt.hp.choice('stride3_0',[1,2]),
            'channels_0' : hyperopt.hp.choice('channels3_0',conv_net_filters),
            'stride_1' : hyperopt.hp.choice('stride3_1',[1,2]),
            'channels_1' : hyperopt.hp.choice('channels3_1',conv_net_filters),
            'stride_2' : 1,
            'channels_2' : hyperopt.hp.choice('channels3_2',conv_net_filters),
        },
        {
            'depth': 4,
            'stride_0' : hyperopt.hp.choice('stride4_0',[1,2]),
            'channels_0' : hyperopt.hp.choice('channels4_0',conv_net_filters),
            'stride_1' : hyperopt.hp.choice('stride4_1',[1,2]),
            'channels_1' : hyperopt.hp.choice('channels4_1',conv_net_filters),
            'stride_2' : 1,
            'channels_2' : hyperopt.hp.choice('channels4_2',conv_net_filters),
            'stride_3' : 1,
            'channels_3' : hyperopt.hp.choice('channels4_3',conv_net_filters),
        },
        ])
    }
best = hyperopt.fmin(lambda c: cross_validated_run(c, n_splits=2), search_space, algo=hyperopt.tpe.suggest, max_evals=HYPER_OPT_EVALS, trials=trials)
with open(f'{BASE_DIR}/best.json', 'w') as f:
    json.dump(hyperopt.space_eval(search_space, best), f, indent=4,sort_keys=True)
with open(f'{BASE_DIR}/trials.pkl', 'wb') as f:
    pickle.dump(trials, f)
