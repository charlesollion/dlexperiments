import torch
import numpy as np
import os
from torchvision import transforms, datasets
from torch.utils.data.dataloader import default_collate


def make_dataloaders(dataset, batch_size, device, subset=None, flatten=True, binarize=True, **kwargs):
    '''
    Build dataloaders for different datasets. The dataloader can be easily iterated on.
    Supports Mnist, FashionMNIST, more to come
    '''

    def collate_fn_spatial(x):
        batch = default_collate(x)[0]
        if binarize:
            batch = torch.distributions.bernoulli.Bernoulli(batch).sample()
        return batch.to(device)

    def collate_fn_flatten(x):
        batch = default_collate(x)[0]
        batch = batch.view(batch.shape[0], -1)
        if binarize:
            batch = torch.distributions.bernoulli.Bernoulli(batch).sample()
        return batch.to(device)

    collate_fn = collate_fn_flatten if flatten else collate_fn_spatial

    if dataset == 'mnist':
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
        val_dataset = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())
        if subset:
            train_dataset = torch.utils.data.Subset(train_dataset, list(range(subset)))
            val_dataset = torch.utils.data.Subset(val_dataset, list(range(subset)))
    elif dataset == 'fashionmnist':
        train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transforms.ToTensor())
        val_dataset = datasets.FashionMNIST('./data', train=False, transform=transforms.ToTensor())
        if subset:
            train_dataset = torch.utils.data.Subset(train_dataset, list(range(subset)))
            val_dataset = torch.utils.data.Subset(val_dataset, list(range(subset)))
    elif dataset == 'cifar':
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
        val_dataset = datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor())
        if subset:
            train_dataset = torch.utils.data.Subset(train_dataset, list(range(subset)))
            val_dataset = torch.utils.data.Subset(val_dataset, list(range(subset)))
    else:
        raise ValueError(dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=batch_size, collate_fn=collate_fn, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=batch_size, collate_fn=collate_fn, shuffle=False, **kwargs)

    return train_loader, val_loader


def save_model(model, name):
    dir = "./checkpoints"
    if not os.path.exists(dir):
        os.mkdir(dir)
    path = os.path.join(dir, name)
    torch.save(model, path)


class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

args = DotDict()


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
