import os
from ray import tune
from cvae_trainable import CVAE_trainable
from load_data import get_mnist_data, get_experimental_data

if __name__ == "__main__":
    # Define constants
    BATCH_SIZE = 100
    epochs = 20
    min_epochs = 5
    BINARIZATION = False
    DATA_TYPE = 'experiment'
    TEST_CANDIDATES = 20
    BASE_DIR = './ray_tune/'
    os.makedirs(BASE_DIR, exist_ok=True)
    if DATA_TYPE=='mnist':
        get_mnist_data(BATCH_SIZE, False)
    elif DATA_TYPE=='experiment':
        get_experimental_data(BATCH_SIZE, os.getcwd())
    else:
        raise RuntimeError(f"Did not understand DATA_TYPE: {DATA_TYPE}")
    conv_net_filters = ['8','16','32'] # Sadly have to store them as strings to cheat and register this in tensorboard
    search_space = {
        'DATA_TYPE': DATA_TYPE,
        'project_dir': os.getcwd(), # Required, because otherwise finding the data is painful.
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
    ahb = tune.schedulers.AsyncHyperBandScheduler(
            time_attr="training_iteration",
            metric="test_loss",
            mode="min",
            grace_period=min_epochs,
            max_t=epochs)
    tune.run(
        CVAE_trainable,
        stop={"training_iteration": epochs},
        verbose=0,
        num_samples=TEST_CANDIDATES,
        local_dir=BASE_DIR,
        resources_per_trial={'gpu': 1},
        scheduler=ahb,
        config=search_space)
