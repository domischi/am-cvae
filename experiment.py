## The following code is heavily based on Google's CVAE tutorial (https://www.tensorflow.org/tutorials/generative/cvae)
## and on Keras' blog post about auto-encoders (https://blog.keras.io/building-autoencoders-in-keras.html)

import tensorflow as tf
import tensorflow.keras as keras
import os
import time
import numpy as np
import sklearn.model_selection
import hyperopt
import json
import pickle
from illustration import *
from load_data import get_mnist_data
from cvae import cross_validated_run


# Define constants
BATCH_SIZE = 100
epochs = 4
save_freq = 2
BINARIZATION = False
HYPER_OPT_EVALS = 1
BASE_DIR = 'data/'
os.makedirs(BASE_DIR, exist_ok=True)


# Define the search space
trials = hyperopt.Trials()
conv_net_filters = [8,16,32]
search_space = {
    'BATCH_SIZE': BATCH_SIZE, 
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

# Do the fitting
best = hyperopt.fmin(lambda c: cross_validated_run(c, *get_mnist_data(BATCH_SIZE, BINARIZATION), n_splits=2), search_space, algo=hyperopt.tpe.suggest, max_evals=HYPER_OPT_EVALS, trials=trials)

# Save the results
timestamp = int( time.time() )
with open(f'{BASE_DIR}/best-{timestamp}.json', 'w') as f:
    json.dump(hyperopt.space_eval(search_space, best), f, indent=4,sort_keys=True)
with open(f'{BASE_DIR}/trials-{timestamp}.pkl', 'wb') as f:
    pickle.dump(trials, f)
