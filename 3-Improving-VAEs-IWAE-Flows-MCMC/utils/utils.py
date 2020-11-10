import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from tqdm.auto import tqdm


def plot_digit_samples(original, reconstucted, generated):
    """
    Plot samples from the generative network in a grid
    """

    grid_h = 2
    grid_w = 5
    data_h = 28
    data_w = 28
    data_c = 1
    plt.close()
    fig, ax = plt.subplots(ncols=3, figsize=(7, 4), dpi=200)
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


def plot_posterior(data, names=None):
    plt.close()
    # import pdb
    # pdb.set_trace()
    labels_np = np.array([float(val) for val in data[0]])
    points_np = data[1]
    n_cols = 2
    fig, ax = plt.subplots(ncols=n_cols, figsize=(6, 6 // n_cols), dpi=200)
    limx, limy = 90, 90
    #### TSNE ####
    tsne = TSNE(n_components=2)
    compressed_data = tsne.fit_transform(points_np)
    ax[0].set_title('TSNE')
    for label in np.unique(labels_np):
        ind = np.argwhere(labels_np == label).squeeze()
        zero = np.zeros_like(ind)
        if names:
            label = names[int(label)]
        ax[0].scatter(compressed_data[ind, zero], compressed_data[ind, zero + 1], alpha=0.25, label=label)
    ax[0].set_xlim((-limx, limx))
    ax[0].set_ylim((-limy, limy))
    ax[0].set_xticks(np.arange(-limx, limx, 25))
    ax[0].set_yticks(np.arange(-limy, limy, 25))
    ax[0].legend(loc=2, prop={'size': 4})
    ax[0].set_aspect('equal', 'box')

    limx, limy = 4, 4
    ### as it is ###
    ax[1].set_title('Posterior samples')
    for label in np.unique(labels_np):
        ind = np.argwhere(labels_np == label).squeeze()
        zero = np.zeros_like(ind)
        ax[1].scatter(points_np[ind, zero], points_np[ind, zero + 1], alpha=0.25, label=label)
    ax[1].set_xlim((-limx, limx))
    ax[1].set_ylim((-limy, limy))
    ax[1].set_xticks(np.arange(-limx, limx, 1))
    ax[1].set_yticks(np.arange(-limy, limy, 1))
    # ax[1].legend()
    ax[1].set_aspect('equal', 'box')
    plt.tight_layout()
    plt.show()


def sample_distr(model, batch):
    with torch.no_grad():
        mu, logvar = model.encode(batch)
        z = model.reparameterize(mu, logvar)
        if hasattr(model, "Flow"):
            z_transformed, log_jac = model.Flow(z)
        else:
            z_transformed = z
            log_jac = torch.zeros_like(z_transformed[:, 0])
        Q_logprob = torch.distributions.Normal(loc=mu, scale=torch.exp(0.5 * logvar)).log_prob(z).sum(-1) - log_jac

    return z_transformed, Q_logprob


def estimate_ll(model, dataset, S=5000, n_runs=3):
    final_nll = []
    for i in range(n_runs):
        model.eval()
        mean_nll = []
        with torch.no_grad():
            for batch, _ in tqdm(dataset):
                batch = batch.view(-1, 784).repeat(S, 1).to(model.device)
                z, Q_logprob = sample_distr(model, batch)
                P = torch.distributions.Normal(loc=torch.zeros_like(z), scale=torch.ones_like(z))
                x_hat = model.decode(z)
                BCE = F.binary_cross_entropy_with_logits(x_hat, batch,
                                                         reduction='none')
                current_nll = torch.logsumexp(
                    -BCE.sum(-1).view((S, -1)) + P.log_prob(z).sum(-1).view((S, -1)) - Q_logprob.view((S, -1)),
                    dim=0).mean() - torch.log(torch.tensor(S, dtype=torch.float32, device=model.device))
                mean_nll.append(-current_nll.cpu().item())
        final_nll.append(np.mean(mean_nll))
    return final_nll
