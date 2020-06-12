import tensorflow as tf
import tensorflow.keras as keras
import os
import numpy as np
from load_data import get_mnist_data
from ray import tune
from cvae import CVAE

## Get rid of some very verbose logging of TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class CVAE_trainable(tune.Trainable):
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
