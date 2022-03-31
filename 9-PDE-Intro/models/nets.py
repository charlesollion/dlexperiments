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
        self.layers = [nn.Dense(feat) for feat in self.features]

    def __call__(self, t, x):
        input = np.concatenate((t,x))
        final_output = 0 # Initialisation
        for i, layer in enumerate(self.layers):
            output = layer(input)
            if i != len(self.layers) - 1:
                input = nn.tanh(output)
            else:
                final_output = output
        return final_output
