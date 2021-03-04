import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.utils import logprob_normal, view_flat, bce, mse
from models.nets import SimpleNN, ConvNN
from models.VQ import VQEmbedding, VQEmbeddingGumbel

class VanillaVAE(nn.Module):
    '''
    Vanilla VAE. Kept as simple as possible
    '''
    def __init__(self, latent_dim, activation = "relu", hidden_dim = 256, output_dim = 784, archi="basic", data_type="binary", split=True):
        super().__init__()
        if archi == "basic":
            self.net = SimpleNN(latent_dim, activation, hidden_dim, output_dim, large=False, split=split)
        elif archi == "large":
            self.net = SimpleNN(latent_dim, activation, hidden_dim, output_dim, large=True, split=split)
        elif archi == "convCifar":
            self.net = ConvNN(latent_dim, activation, hidden_dim, split=split)
        elif archi == "convMnist":
            self.net = ConvNN(latent_dim, activation, hidden_dim, image_size = (28,28), channel_num=1, num_layers=2, split=split)
        else:
            raise ValueError("Architecture unknown: " + str(archi))
        self.latent_dim = latent_dim
        self.description = "vae"
        self.data_type = data_type
        if data_type == "binary":
            self.reconstruction_loss = bce
        elif data_type == "continuous":
            self.reconstruction_loss = mse


    def encode(self, x):
        return self.net.encode(x)


    def sample(self, mu, logvar):
        # Reparametrization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def decode(self, z):
        return self.net.decode(z)


    def log_prob(self, z, x=None):
        log_likelihood = - self.reconstruction_loss(self.decode(z), x)
        log_prior = torch.distributions.Normal(loc=torch.tensor(0., device=z.device, dtype=torch.float32),
                                               scale=torch.tensor(1., device=z.device, dtype=torch.float32)).log_prob(z).sum(-1)
        return log_likelihood + log_prior


    def forward(self, z):
        if self.data_type== "binary":
            return torch.sigmoid(self.decode(z))
        else:
            return self.decode(z)


    def loss_function(self, x_rec, x, mu, logvar):
        BCE = self.reconstruction_loss(x_rec, x).mean()
        KLD = -0.5 * torch.mean((1 + logvar - mu.pow(2) - logvar.exp()).sum(-1))
        loss = BCE + KLD
        return loss, BCE


    def step(self, x):
        # Computes the loss from a batch of data
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        x_rec = self.decode(z)
        loss, BCE = self.loss_function(view_flat(x_rec), view_flat(x), mu, logvar)
        return loss, x_rec, z, BCE


    def get_optimizer(self, lr=1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)


    def importance_distr(self, x):
        mu, logvar = self.encode(x)
        return torch.distributions.Normal(loc=mu, scale=torch.exp(0.5 * logvar))



class IWAE(VanillaVAE):
    '''
    Importance Weighted Auto Encoder
    Basis taken from Nikita's code
    '''
    def __init__(self, num_samples, latent_dim, activation="relu", clamp=1e6, hidden_dim = 256, output_dim = 784, archi="basic", data_type="binary"):
        super().__init__(latent_dim, activation, hidden_dim, output_dim, archi, data_type)
        self.num_samples = num_samples
        self.clamp_kl = clamp
        self.description = f"iwae n={num_samples}"


    def sample(self, mu, logvar):
        ''' Reparametrization trick
        Will output a tensor of shape ``(num_sample, batch_size, [input_dim])`
        '''
        std = torch.exp(0.5 * logvar)
        # Repeat the std along the first axis to sample multiple times
        dims = (self.num_samples,) + (std.shape)
        eps = torch.randn(*dims, device=mu.device)
        return mu + eps * std


    def loss_function(self, x_rec, x, mu, logvar, z):
        log_Q = logprob_normal(z,mu,logvar).view((self.num_samples, -1, self.latent_dim)).sum(-1)
        log_Pr = (-0.5 * z ** 2).view((self.num_samples, -1, self.latent_dim)).sum(-1)
        KL_eq = torch.clamp(log_Q - log_Pr, -self.clamp_kl, self.clamp_kl)
        BCE = self.reconstruction_loss(x_rec, x)

        log_weight = - BCE - KL_eq
        log_weight = log_weight - torch.max(log_weight, 0)[0]  # for stability
        weight = torch.exp(log_weight)
        weight = weight / torch.sum(weight, 0)
        weight = weight.detach()
        loss = torch.mean(torch.sum(weight * (BCE + KL_eq), 0))

        return loss, torch.sum(BCE * weight, dim=0).mean()


    def step(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        x_rec = self.decode(z)
        loss, BCE = self.loss_function(view_flat(x_rec), view_flat(x), mu, logvar, z)
        return loss, x_rec, z, BCE



class VectorQuantizedVAE(VanillaVAE):
    """Vector Quantized VAE
    latent_dim: embedding (codebook) vector length
    K: codebook size (number of discrete values)
    gumbel: if True, uses a gumbel softmax instead of the straight through operator
    """
    def __init__(self, latent_dim, K=10, gumbel=False, tau=1., beta=1., activation="relu", hidden_dim = 256, output_dim = 784, archi="basic", data_type="binary"):
        super().__init__(latent_dim, activation, hidden_dim, output_dim, archi, data_type, split=False)
        self.description = f"vq-vae K={K}"
        self.gumbel = gumbel
        self.beta = beta
        self.tau = tau
        if gumbel:
            self.codebook = VQEmbeddingGumbel(K, latent_dim, self.tau)
        else:
            self.codebook = VQEmbedding(K, latent_dim)


    def encode(self, x):
        z_e_x = self.net.encode(x)
        latents = self.codebook(z_e_x)
        return latents


    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents)
        x_tilde = self.net.decode(z_q_x)
        return x_tilde


    def forward(self, x):
        z_e_x = self.net.encode(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.net.decode(z_q_x_st)
        return x_tilde, z_e_x, z_q_x


    def step(self, x):
        # Computes the loss from a batch of data
        x_tilde, z_e_x, z_q_x = self.forward(x)

        # Reconstruction loss
        loss_recons = self.reconstruction_loss(x_tilde, x).mean()


        if self.gumbel:
            # Vector quantization objective
            loss_vq = - F.cosine_similarity(z_q_x, z_e_x.detach()).mean()
            # Commitment objective
            loss_commit = - F.cosine_similarity(z_e_x, z_q_x.detach()).mean()
            loss = loss_recons + loss_vq + self.beta * loss_commit
        else:
            # Vector quantization objective
            loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
            # Commitment objective
            loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
            loss = loss_recons + loss_vq + self.beta * loss_commit
        return loss, x_tilde, z_e_x, loss_recons
