import matplotlib.pyplot as plt
import ncvis
import numpy as np
from sklearn.manifold import TSNE


def plot_digit_samples(original, reconstucted, generated):
    """
    Plot samples from the generative network in a grid
    """

    grid_h = 8
    grid_w = 8
    data_h = 28
    data_w = 28
    data_c = 1
    plt.close()
    fig, ax = plt.subplots(ncols=3)
    images_list = [original, reconstucted, generated]
    names = ['original', 'reconstructed', 'generated']
    for pos in range(3):
        # Turn the samples into one large image
        tiled_img = np.zeros((data_h * grid_h, data_w * grid_w))

        for idx, image in enumerate(images_list[pos]):
            i = idx % grid_w
            j = idx // grid_w

            top = j * data_h
            bottom = (j + 1) * data_h
            left = i * data_w
            right = (i + 1) * data_w
            tiled_img[top:bottom, left:right] = image

        # Plot the new image
        ax[pos].set_title(names[pos])
        ax[pos].axis('off')
        ax[pos].imshow(tiled_img, cmap='gray')
    plt.tight_layout()
    plt.show()


def plot_posterior(data):
    plt.close()

    labels_np = np.array([float(val) for val in data[0]])
    points_np = data[1]

    fig, ax = plt.subplots(ncols=2, figsize=(6, 3), dpi=200)
    limx, limy = 90, 90
    #### TSNE ####
    tsne = TSNE(n_components=2)
    compressed_data = tsne.fit_transform(points_np)
    ax[0].set_title('TSNE')
    for label in np.unique(labels_np):
        ind = np.argwhere(labels_np == label).squeeze()
        zero = np.zeros_like(ind)
        ax[0].scatter(compressed_data[ind, zero], compressed_data[ind, zero + 1], alpha=0.25, label=label)
    ax[0].set_xlim((-limx, limx))
    ax[0].set_ylim((-limy, limy))
    ax[0].set_xticks(np.arange(-limx, limx, 25))
    ax[0].set_yticks(np.arange(-limy, limy, 25))
    # ax[0].legend()
    ax[0].set_aspect('equal', 'box')

    limx, limy = 35, 35
    ### NCVIS ###
    vis = ncvis.NCVis()
    compressed_data = vis.fit_transform(points_np)
    ax[1].set_title('NCVis')
    for label in np.unique(labels_np):
        ind = np.argwhere(labels_np == label).squeeze()
        zero = np.zeros_like(ind)
        ax[1].scatter(compressed_data[ind, zero], compressed_data[ind, zero + 1], alpha=0.25, label=label)
    ax[1].set_xlim((-limx, limx))
    ax[1].set_ylim((-limy, limy))
    ax[1].set_xticks(np.arange(-limx, limx, 25))
    ax[1].set_yticks(np.arange(-limy, limy, 25))
    # ax[1].legend()
    ax[1].set_aspect('equal', 'box')
    plt.tight_layout()
    plt.show()
