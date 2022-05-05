from turtle import shape
import jax.numpy as np
import jax
from jax import grad, jit, vmap, jacfwd, jacrev
from jax import random
import flax
from flax import linen as nn
from flax.core import freeze, unfreeze
from typing import Sequence

class MLP(nn.Module):
    features: Sequence[int]

    def setup(self):
        self.layers_m1 = [nn.Dense(feat) for feat in self.features]
        self.layers_m2 = [nn.Dense(feat) for feat in self.features]

    def __call__(self, t, x):
        input = np.concatenate((t,x))
        final_output = 0 # Initialisation
        for i, (layer1, layer2) in enumerate(zip(self.layers_m1, self.layers_m2)):
            x1 = layer1(input)
            x2 = layer2(input)
            output = (1. - t) * x1 + t * x2
            if i != len(self.layers_m1) - 1:
                input = nn.relu(output)
            else:
                final_output = output
        return final_output

class UNet(nn.Module):
    features: Sequence[int]

    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]
        self.layer_m1 = [nn.Dense(feat) for feat in self.features]
        self.layer_m2 = [nn.Dense(feat) for feat in self.features]

    def mix(self, t, x):
        x1 = self.layer_m1(x)
        x2 = self.layer_m2(x)
        return (1. - t) * x1 + t * x2

    def __call__(self, t, x):
        input = np.concatenate((t,x))
        final_output = 0 # Initialisation
        for i, layer in enumerate(self.layers):
            output = layer(input)
            if i != len(self.layers) - 1:
                input = nn.relu(output)
            else:
                final_output = output
        return final_output
