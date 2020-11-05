import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.normflows import NormFlow
from models.samplers import HMC


class Base(pl.LightningModule):
    def __init__(self, act_func, num_samples, hidden_dim, name="VAE"):
        super(Base, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, hidden_dim)
        self.fc22 = nn.Linear(400, hidden_dim)
        # Decoder
        self.fc3 = nn.Linear(hidden_dim, 400)
        self.fc4 = nn.Linear(400, 784)

        self.act_func = act_func
        self.num_samples = num_samples
        # Fixed random vector, which we recover each epoch
        self.random_z = torch.randn((64, hidden_dim), dtype=torch.float32)

        self.name = name

    def encode(self, x):
        h1 = self.act_func(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = self.act_func(self.fc3(z))
        return self.fc4(h3)

    def forward(self, z):
        return self.decode(z)

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        x_hat = torch.sigmoid(self(self.random_z.to(val_loss.device))).view((-1, 1, 28, 28))
        grid = torchvision.utils.make_grid(x_hat).mean(0, keepdim=True)

        self.logger.experiment.add_image(f'image/{self.name}', grid, self.current_epoch)
        self.logger.experiment.add_scalar(f'avg_val_loss/{self.name}', val_loss, self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        return {"loss": self.step(batch)[0]}

    def validation_step(self, batch, batch_idx):
        return {"val_loss": self.step(batch)[0]}


class VAE(Base):
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy_with_logits(recon_x, x.view(-1, 784), reduction='none').view(
            (self.num_samples, -1, 784)).mean(0).sum(-1).mean()
        KLD = -0.5 * torch.mean((1 + logvar - mu.pow(2) - logvar.exp()).view(
            (self.num_samples, -1, self.fc22.out_features)).mean(0).sum(-1))
        loss = BCE + KLD
        return loss

    def step(self, batch):
        x, _ = batch
        x = x.repeat(self.num_samples, 1, 1, 1)
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        x_hat = self(z)
        loss = self.loss_function(x_hat, x, mu, logvar)
        return loss, x_hat, z


class IWAE(Base):
    def loss_function(self, recon_x, x, mu, logvar, z):
        log_Q = torch.distributions.Normal(loc=mu,
                                           scale=torch.exp(0.5 * logvar)).log_prob(z).view(
            (self.num_samples, -1, self.fc22.out_features)).sum(-1)

        log_Pr = torch.sum((-0.5 * z ** 2).view((self.num_samples, -1, self.fc22.out_features)), -1)
        BCE = F.binary_cross_entropy_with_logits(recon_x, x.view(-1, 784), reduction='none').view(
            (self.num_samples, -1, 784)).sum(-1)
        log_weight = log_Pr + BCE - log_Q
        log_weight = log_weight - torch.max(log_weight, 0)[0]  # for stability
        weight = torch.exp(log_weight)
        weight = weight / torch.sum(weight, 0)
        weight = weight.detach()
        loss = torch.mean(torch.sum(weight * (-log_Pr + BCE + log_Q), 0))

        return loss

    def step(self, batch):
        x, _ = batch
        x = x.repeat(self.num_samples, 1, 1, 1)
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        x_hat = self(z)
        loss = self.loss_function(x_hat, x, mu, logvar, z)
        return loss, x_hat, z


class VAE_with_flows(Base):
    def __init__(self, act_func, num_samples, hidden_dim, flow_type, num_flows, need_permute):
        super(VAE_with_flows, self).__init__(act_func, num_samples, hidden_dim)
        self.Flow = NormFlow(flow_type, num_flows, hidden_dim, need_permute)
        self.name += f'_{flow_type}_{num_flows}'

    def loss_function(self, recon_x, x, mu, logvar, z, z_transformed, log_jac):
        BCE = F.binary_cross_entropy_with_logits(recon_x, x.view(-1, 784), reduction='none').view(
            (self.num_samples, -1, 784)).sum(-1).mean()
        log_Q = torch.distributions.Normal(loc=mu, scale=torch.exp(0.5 * logvar)).log_prob(z)
        log_Pr = (-0.5 * z_transformed ** 2)
        KLD = torch.mean((log_Q - log_Pr).view(
            (self.num_samples, -1, self.fc22.out_features)).mean(0).sum(-1) - log_jac.view((self.num_samples, -1)).mean(
            0))
        loss = BCE + KLD
        return loss

    def step(self, batch):
        x, _ = batch
        x = x.repeat(self.num_samples, 1, 1, 1)
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        z_transformed, log_jac = self.Flow(z)
        x_hat = self(z_transformed)
        loss = self.loss_function(x_hat, x, mu, logvar, z, z_transformed, log_jac)
        return loss, x_hat, z_transformed


class VAE_MCMC(Base):
    def __init__(self, act_func, num_samples, hidden_dim, n_leapfrogs, step_size, use_barker):
        super(VAE_MCMC, self).__init__(act_func, num_samples, hidden_dim, name=f"VAE_MCMC_{n_leapfrogs}")
        self.sampler = HMC(n_leapfrogs=n_leapfrogs, step_size=step_size, use_barker=use_barker)

    def loss_function(self, recon_x, x, mu, logvar, inference_part=False):
        if inference_part:
            BCE = F.binary_cross_entropy_with_logits(recon_x, x.view(-1, 784), reduction='none').view(
                (self.num_samples, -1, 784)).mean(0).sum(-1).mean()
            KLD = -0.5 * torch.mean((1 + logvar - mu.pow(2) - logvar.exp()).view(
                (self.num_samples, -1, self.fc22.out_features)).mean(0).sum(-1))
            loss = BCE + KLD
        else:
            BCE = F.binary_cross_entropy_with_logits(recon_x, x.view(-1, 784), reduction='none').view(
                (self.num_samples, -1, 784)).mean(0).sum(-1).mean()
            loss = BCE
        return loss

    def joint_density(self, ):
        def density(z, x):
            x_reconst = self(z)
            log_Pr = torch.distributions.Normal(loc=torch.tensor(0., device=z.device, dtype=torch.float32),
                                                scale=torch.tensor(1., device=z.device, dtype=torch.float32)).log_prob(
                z).sum(-1)
            return -F.binary_cross_entropy_with_logits(x_reconst, x.view(-1, 784), reduction='none').sum(-1) + log_Pr

        return density

    def step(self, batch):
        x, _ = batch
        x = x.repeat(self.num_samples, 1, 1, 1)
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        with torch.no_grad():
            x_hat = self(z)
        loss_1 = self.loss_function(x_hat, x, mu, logvar, inference_part=True)

        z_transformed = self.sampler.run_chain(z_init=z.detach(), target=self.joint_density(), x=x, n_steps=10)
        x_hat = self(z_transformed)
        loss_2 = self.loss_function(x_hat, x, mu, logvar)

        loss = loss_1 + loss_2

        return loss, x_hat, z_transformed
