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
        h = np.concatenate((t,x))
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = nn.tanh(h)
        return h
