import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from tqdm.auto import tqdm

from models.samplers import HMC


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


def sample_distr(model, batch, sampler):
    with torch.no_grad():
        mu, _ = model.encode(batch.view(-1, 784))
        if hasattr(model, "Flow"):
            mu = model.Flow(mu)[0]
        z_transformed = sampler.run_chain(z_init=mu.detach(), target=model.joint_density(), x=batch, n_steps=200,
                                          return_trace=True, burnin=100)
        mu = z_transformed.view((200, -1, mu.shape[-1])).mean(0)
        scale = 1.2 * z_transformed.view((200, -1, mu.shape[-1])).std(0)
    return mu, scale


def estimate_ll(model, dataset, S=1000):
    model.eval()
    mean_nll = []
    sampler = HMC(n_leapfrogs=5, step_size=0.005, use_barker=True).to(model.device)
    with torch.no_grad():
        for batch, _ in tqdm(dataset):
            batch = batch.to(model.device)
            mu, scale = sample_distr(model, batch, sampler)
            Q = torch.distributions.Normal(loc=mu, scale=scale)
            P = torch.distributions.Normal(loc=torch.zeros_like(mu), scale=torch.ones_like(scale))
            z = Q.sample((S,))
            x_hat = model.decode(z)
            BCE = F.binary_cross_entropy_with_logits(x_hat, batch.view(-1, 784).repeat(S, 1).view((S, -1, 784)),
                                                     reduction='none').view(
                (S, -1, 784)).mean(0).sum(-1)
            current_nll = torch.logsumexp(
                -BCE + P.log_prob(z).mean(0).sum(-1) - Q.log_prob(z).mean(0).sum(-1), dim=0) - torch.log(
                torch.tensor(S, dtype=torch.float32, device=model.device))
            mean_nll.append(current_nll.cpu().item())
    return mean_nll
