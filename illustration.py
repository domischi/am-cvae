import matplotlib.pyplot as plt
import numpy as np

def generate_and_save_examples(model, epoch, test_input, BASE_DIR='./'):
    predictions = model.sample(test_input)
    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.savefig(f'{BASE_DIR}/examples_at_epoch_{epoch:04d}.png')


def plot_digit_classes(cvae,
                       x_test,
                       epoch, BASE_DIR='./'):
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

def plot_digits_over_latent(cvae,
                            epoch,n=10, BASE_DIR='./'):
    if cvae.latent_dim != 2:
        return
    filename = f"{BASE_DIR}/digits_over_latent_{epoch:04d}.png"
    # display a 10x10 2D manifold of digits
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

def plot_results(cvae, data, rand_vec, epoch, BASE_DIR):
    generate_and_save_examples(model, epoch, rand_vec, BASE_DIR=BASE_DIR)
    plot_digit_classes(cvae, data, epoch, BASE_DIR=BASE_DIR)
    plot_digits_over_latent(cvae, epoch, BASE_DIR=BASE_DIR)
