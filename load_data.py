import tensorflow as tf

# Load the MNIST dataset
def get_mnist_data(BATCH_SIZE, BINARIZATION):
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

    # Binarization
    if BINARIZATION:
        train_images[train_images >= .5] = 1.
        train_images[train_images < .5] = 0.
        test_images[test_images >= .5] = 1.
        test_images[test_images < .5] = 0.
    return train_images, test_images
