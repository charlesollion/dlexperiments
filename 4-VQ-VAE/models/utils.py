import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_crossentropy_logits_stable(x, y):
    '''
    Special binary crossentropy where y can be a single data batch, while
    x has several repeats
    '''
    return torch.clamp(x, 0) - x * y + torch.log(1 + torch.exp(-torch.abs(x)))


def mse(x, y):
    return torch.square(x-y).sum(-1)


def bce(x,y):
    return binary_crossentropy_logits_stable(x,y).sum(-1)


def logprob_normal(z, mu=torch.tensor(0.), logvar=torch.tensor(0.)):
    '''
    Similar to torch.distributions.Normal(mu, exp(logvar * 0.5)).log_prob(z)
    '''
    return -(z - mu) ** 2 / (2 * torch.exp(logvar)) - 0.5 * logvar - 0.919


def view_flat(x):
    '''
    flattens the spatial dimensions if any
    x is a tensor of various input shape
    '''
    s = x.shape
    if len(s) < 4:
        # no spatial dimensions
        return x
    if len(s) == 4:
        # (batch, channels, h, w)
        return x.view(s[0], -1)
    elif len(s) == 5:
        # (nsamples, batch, chanels, h, w)
        return x.view(s[0], s[1], -1)
