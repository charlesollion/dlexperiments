from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms, datasets

from models import VAE, IWAE, VAE_with_flows, VAE_MCMC


def get_activations():
    return {
        "relu": torch.nn.ReLU,
        "leakyrelu": torch.nn.LeakyReLU,
        "tanh": torch.nn.Tanh,
    }


def make_dataloaders(dataset, gpus, batch_size, val_batch_size):
    kwargs = {'num_workers': 20, 'pin_memory': True} if gpus else {}
    if dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
            batch_size=val_batch_size, shuffle=False, **kwargs)

    elif dataset == 'fashionmnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data', train=True, download=True,
                                  transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data', train=False, transform=transforms.ToTensor()),
            batch_size=val_batch_size, shuffle=False, **kwargs)
    else:
        raise ValueError

    return train_loader, val_loader


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    tb_logger = pl_loggers.TensorBoardLogger('lightning_logs/')
    parser.add_argument("--model", default="VAE_with_flows", choices=["VAE", "IWAE", "VAE_with_flows", "VAE_MCMC"])
    parser.add_argument("--flow_type", default="None", choices=["IAF", "BNAF", "RNVP", "None"])
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--val_batch_size", default=50, type=int)
    parser.add_argument("--hidden_dim", default=2, type=int)
    parser.add_argument("--need_permute", default=False, type=bool)
    parser.add_argument("--n_leapfrogs", default=5, type=int)
    parser.add_argument("--step_size", default=0.05, type=float)
    parser.add_argument("--use_barker", default=True, type=bool)

    parser.add_argument("--num_samples", default=1, type=int)
    parser.add_argument("--num_flows", default=5, type=int)
    parser.add_argument("--act_func", default="tanh", choices=["relu", "leakyrelu", "tanh"])
    act_func = get_activations()

    parser.add_argument("--dataset", default='fashionmnist', choices=['mnist', 'fashionmnist'])

    args = parser.parse_args()
    args.gpus = 1

    train_loader, val_loader = make_dataloaders(dataset=args.dataset,
                                                gpus=args.gpus,
                                                batch_size=args.batch_size,
                                                val_batch_size=args.val_batch_size)
    if args.model == "VAE":
        model = VAE(act_func=act_func[args.act_func], num_samples=args.num_samples, hidden_dim=args.hidden_dim)
    elif args.model == "IWAE":
        model = IWAE(act_func=act_func[args.act_func], num_samples=args.num_samples, hidden_dim=args.hidden_dim,
                     name=args.model)
    elif args.model == "VAE_with_flows":
        model = VAE_with_flows(act_func=act_func[args.act_func], num_samples=args.num_samples, num_flows=args.num_flows,
                               hidden_dim=args.hidden_dim, flow_type=args.flow_type, need_permute=args.need_permute)
    elif args.model == "VAE_MCMC":
        model = VAE_MCMC(act_func=act_func[args.act_func], num_samples=args.num_samples, hidden_dim=args.hidden_dim,
                         n_leapfrogs=args.n_leapfrogs, step_size=args.step_size, use_barker=args.use_barker)
    else:
        raise ValueError

    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger, deterministic=True, max_epochs=150)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
    trainer.save_checkpoint(f"./checkpoints/{args.model}_{args.flow_type}.ckpt")
