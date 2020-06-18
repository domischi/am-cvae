import tensorflow as tf
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import os

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

# Prepare the experimental pictures
def resize_experimental_data(data_dir='./data', raw_folder = 'raw', larger_pixel_size = 32):
    for T in ['train', 'test']:
        file_list = glob(f"{data_dir}/{raw_folder}/{T}/**/*.tif", recursive=True)
        assert(len(file_list)>0 or T=='test') # We might want to train with all data (i.e. having no out of sample test data)
        im = Image.open(file_list[0])
        h = im.height
        w = im.width
        assert(np.isclose(w/h,1920/1200)) ## aspect ratio 8/5
        smaller_pixel_size = int(larger_pixel_size * h/w)
        save_dir = f"{data_dir}/{larger_pixel_size}x{smaller_pixel_size}/"
        img_dir = f'{save_dir}/{T}_img'
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        data = []
        for f in tqdm(file_list):
            save_name = '_'.join(f.split('/')[3:])
            im = Image.open(f)
            if im.mode =='I;16':
                im = im.point(lambda i:i*(1./256)).convert('L')
            assert(im.mode == 'L')
            try:
                im_resized = im.resize((larger_pixel_size,smaller_pixel_size))
            except ValueError:
                print(im)
                continue
            im_resized.save(f'{img_dir}/{save_name}')
            data.append(np.asarray(im_resized))
        data = np.array(data) ## shape: (data_size, h, w)
        data = np.transpose(data,axes=[0,2,1]) ## shape: (data_size, h, w)
        with open(f'{save_dir}/{T}.npy', 'wb') as np_file:
            np.save(np_file,data)

# Load the experimental data into memory (same as what get_mnist_data returns for MNIST)
def get_experimental_data(BATCH_SIZE,project_dir, data_dir='./data', larger_pixel_size = 32):
    # Locate data
    smaller_pixel_size = int(larger_pixel_size * 5/8)
    resized_data_dir = f'{project_dir}/{data_dir}/{larger_pixel_size}x{smaller_pixel_size}'

    # If it doesn't exist already, load it in
    if not os.path.isdir(resized_data_dir):
        print("Need to generate the Tensorflow compatible data from the raw images")
        resize_experimental_data(data_dir=data_dir, larger_pixel_size = larger_pixel_size)
    assert(os.path.isdir(resized_data_dir))
    assert(os.path.isfile(f"{resized_data_dir}/train.npy"))
    assert(os.path.isfile(f"{resized_data_dir}/test.npy"))

    # Load the data
    train_images = np.load(f"{resized_data_dir}/train.npy")
    test_images = np.load(f"{resized_data_dir}/train.npy")

    image_size_x = train_images.shape[1]
    image_size_y = train_images.shape[2]

    train_images = train_images.reshape(train_images.shape[0], image_size_x, image_size_y, 1).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], image_size_x, image_size_y, 1).astype('float32')

    ## We need to ensure that train_images.shape[0] is divisible by BATCH_SIZE due to hardcoded resizes
    train_batch_throwaway = (train_images.shape[0] // BATCH_SIZE)*BATCH_SIZE
    test_batch_throwaway = (test_images.shape[0] // BATCH_SIZE)*BATCH_SIZE
    train_images = train_images[:train_batch_throwaway]
    test_images = test_images[:test_batch_throwaway]
    # Assert that it worked
    assert train_images.shape[0] % BATCH_SIZE == 0
    assert test_images.shape[0] % BATCH_SIZE == 0

    # Normalizing the images to the range of [0., 1.]
    train_images /= 255.
    test_images /= 255.

    return train_images, test_images
    

if __name__ == '__main__':
    get_experimental_data(100)
