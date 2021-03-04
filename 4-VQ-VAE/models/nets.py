import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NeuralNet(nn.Module):
    '''
    Encoder and decoder networks
    '''
    def __init__(self, latent_dim, activation = "relu", hidden_dim = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        if activation == "relu":
            self.act = F.relu
        elif activation == "softplus":
            self.act = F.softplus
        elif activation == "leaky_relu":
            self.act = F.leaky_relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(activation)


    def encode(self, x):
        return x


    def decode(self, z):
        return z



class SimpleNN(NeuralNet):
    def __init__(self, latent_dim, activation = "relu", hidden_dim = 256, output_dim = 784, large=False, split=True):
        super().__init__(latent_dim, activation, hidden_dim)
        self.output_dim = output_dim
        self.split = split

        # Encoder
        self.l1_enc = nn.Linear(output_dim, hidden_dim)
        self.l2_enc = nn.Linear(hidden_dim, 2 * latent_dim if self.split else latent_dim)

        # Decoder
        self.l1_dec = nn.Linear(latent_dim, hidden_dim)
        self.l2_dec = nn.Linear(hidden_dim, output_dim)

        # if "large" network, add an extra hidden layer
        self.extra_hidden_layer = large

        if large:
            self.lh_dec = nn.Linear(hidden_dim, hidden_dim)
            self.lh_enc = nn.Linear(hidden_dim, hidden_dim)


    def encode(self, x):
        # Split mu and logvar
        h = self.act(self.l1_enc(x))
        if self.extra_hidden_layer:
            h = self.act(self.lh_enc(h))
        h = self.l2_enc(h)
        if self.split:
            return torch.split(h, self.latent_dim, -1)
        else:
            return h


    def decode(self, z):
        h = self.act(self.l1_dec(z))
        if self.extra_hidden_layer:
            h = self.act(self.lh_dec(h))
        x = self.l2_dec(h)
        return x



class ConvNN(NeuralNet):
    def __init__(self, latent_dim, activation = "relu", hidden_dim = 128, image_size = (32,32), channel_num=3, num_layers=3):
        super().__init__(latent_dim, activation, hidden_dim)
        self.image_size = image_size
        self.channel_num = channel_num
        self.num_layers = num_layers

        self.feature_volume = image_size[0] // (2**num_layers) * image_size[1] // (2**num_layers) * hidden_dim
        self.inner_spatial_dims = (self.hidden_dim, self.image_size[0] // (2**num_layers), self.image_size[1] // (2**num_layers))

        # Encoder
        if num_layers == 2:
            self.conv1 = self._conv(channel_num, hidden_dim // 2)
            self.conv2 = self._conv(hidden_dim // 2, hidden_dim)
            self.l1_enc = nn.Linear(self.feature_volume, latent_dim * 2)
        elif num_layers == 3:
            self.conv1 = self._conv(channel_num, hidden_dim // 4)
            self.conv2 = self._conv(hidden_dim // 4, hidden_dim // 2)
            self.conv3 = self._conv(hidden_dim // 2, hidden_dim)
            self.l1_enc = nn.Linear(self.feature_volume, latent_dim * 2)

        # Decoder
        if num_layers == 2:
            self.l1_dec = nn.Linear(latent_dim, self.feature_volume)
            self.deconv1 = self._deconv(hidden_dim, hidden_dim // 2)
            self.deconv2 = self._deconv(hidden_dim // 2, channel_num)
        elif num_layers == 3:
            self.l1_dec = nn.Linear(latent_dim, self.feature_volume)
            self.deconv1 = self._deconv(hidden_dim, hidden_dim // 2)
            self.deconv2 = self._deconv(hidden_dim // 2, hidden_dim // 4)
            self.deconv3 = self._deconv(hidden_dim // 4, channel_num)


    def encode(self, x):
        # Split mu and logvar
        h = self.conv1(x)
        h = self.conv2(h)
        if self.num_layers == 3:
            h = self.conv3(h)
        flat_h = self.l1_enc(h.view(-1, self.feature_volume))
        return torch.split(flat_h, self.latent_dim, -1)


    def decode(self, z):
        output_dims = z.shape[0:-1] + (self.channel_num,) + self.image_size
        h = self.l1_dec(z).view(-1, *self.inner_spatial_dims) # (batch * num_samples, c, h, w)
        h = self.deconv1(h)
        h = self.deconv2(h)
        if self.num_layers == 3:
            h = self.deconv3(h) # (batch * num_samples, 3, h, w)
        return h.view(*output_dims) # (num_samples, batch, 3, h, w)


    def _conv(self, channel_size, kernel_num):
        return nn.Sequential(
            nn.Conv2d(
                channel_size, kernel_num,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )


    def _deconv(self, channel_num, kernel_num):
        return nn.Sequential(
            nn.ConvTranspose2d(
                channel_num, kernel_num,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )
