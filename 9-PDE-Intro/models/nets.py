from turtle import shape
import jax.numpy as jnp
import jax
from jax import grad, jit, vmap, jacfwd, jacrev
from jax import random
import flax
from flax import linen as fnn
from flax.core import freeze, unfreeze
from typing import Sequence

class MLP(fnn.Module):
    features: Sequence[int]

    def setup(self):
        self.layers_m1 = [fnn.Dense(feat) for feat in self.features]
        self.layers_m2 = [fnn.Dense(feat) for feat in self.features]

    def __call__(self, t, x):
        input = jnp.concatenate((t,x))
        final_output = 0 # Initialisation
        for i, (layer1, layer2) in enumerate(zip(self.layers_m1, self.layers_m2)):
            x1 = layer1(input)
            x2 = layer2(input)
            output = (1. - t) * x1 + t * x2
            if i != len(self.layers_m1) - 1:
                input = fnn.relu(output)
            else:
                final_output = output
        return final_output

class SinusoidalPosEmb(fnn.Module):
    def __init__(self, dim, num_steps, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)
        self.lin1 = fnn.Dense(dim // 4, dim)
        self.lin2 = fnn.Dense(dim, dim)
        self.act = fnn.relu()

    def __call__(self, t):
        t = t / self.num_steps * self.rescale_steps
        half_dim = self.dim // 8
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = jnp.cat((emb.sin(), emb.cos()), dim=-1)

        # MLP transformation
        out = self.act(self.lin1(emb))
        return self.lin2(out)



class TimeConditionalLinear(fnn.Module):
    def __init__(self, num_in, num_out, n_steps, mode="sinusoidal"):
        super(TimeConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = fnn.Linear(num_in, num_out)
        self.mode = mode
        if mode=="sinusoidal":
            self.time_embed = SinusoidalPosEmb(num_out, n_steps)
        elif mode=="embed":
            self.time_embed = fnn.Embedding(n_steps, num_out)
            self.time_embed.weight.data.uniform_()
        elif mode=="gated":
            self.n_steps = float(n_steps)
            self.lin2 = fnn.Linear(num_in, num_out)
        else:
            raise ValueError(mode)

    def forward(self, x, t):
        out = self.lin(x)
        B, S, H = out.shape
        if self.mode == "sinusoidal":
            return self.time_embed(t).view(B, 1, H) + out
        elif self.mode == "embed":
            return self.time_embed(t).view(B, 1, H) * out
        elif self.mode == "gated":
            out2 = self.lin2(x)
            t = (self.n_steps - t) / self.n_steps
            t = t.view(B, 1, 1)
            return t * out + (1. - t) * out2
        else:
            raise ValueError(self.mode)


class Encoder(fnn.Module):
    features: int = 64
    training: bool = True

    @fnn.compact
    def __call__(self, x, t):
        z1_t = jnp.ones(x.shape) * t
        z1 = fnn.Conv(self.features, kernel_size=(3, 3))(x + z1_t)
        z1 = fnn.relu(z1)
        z1 = fnn.Conv(self.features, kernel_size=(3, 3))(z1)
        #z1 = fnn.BatchNorm(use_running_average=not self.training)(z1)
        z1 = fnn.relu(z1)
        z1_pool = fnn.max_pool(z1, window_shape=(2, 2), strides=(2, 2))

        z2 = fnn.Conv(self.features * 2, kernel_size=(3, 3))(z1_pool)
        z2 = fnn.relu(z2)
        z2 = fnn.Conv(self.features * 2, kernel_size=(3, 3))(z2)
        #z2 = fnn.BatchNorm(use_running_average=not self.training)(z2)
        z2 = fnn.relu(z2)
        z2_pool = fnn.max_pool(z2, window_shape=(2, 2), strides=(2, 2))

        z3 = fnn.Conv(self.features * 4, kernel_size=(3, 3))(z2_pool)
        z3 = fnn.relu(z3)
        z3 = fnn.Conv(self.features * 4, kernel_size=(3, 3))(z3)
        #z3 = fnn.BatchNorm(use_running_average=not self.training)(z3)
        z3 = fnn.relu(z3)
        z3_pool = fnn.max_pool(z3, window_shape=(2, 2), strides=(2, 2))

        z4 = fnn.Conv(self.features * 8, kernel_size=(3, 3))(z3_pool)
        z4_t = jnp.ones(z4.shape) * t
        z4 = fnn.relu(z4 + z4_t)
        z4 = fnn.Conv(self.features * 8, kernel_size=(3, 3))(z4)
        #z4 = fnn.BatchNorm(use_running_average=not self.training)(z4)
        z4 = fnn.relu(z4)
        #z4_dropout = fnn.Dropout(0.5, deterministic=False)(z4)
        z4_pool = fnn.max_pool(z4, window_shape=(2, 2), strides=(2, 2))


        z5 = fnn.Conv(self.features * 16, kernel_size=(3, 3))(z4_pool)
        z5 = fnn.relu(z5)
        z5 = fnn.Conv(self.features * 16, kernel_size=(3, 3))(z5)
        #z5 = fnn.BatchNorm(use_running_average=not self.training)(z5)
        z5 = fnn.relu(z5)
        #z5_dropout = fnn.Dropout(0.5, deterministic=False)(z5)

        return z1, z2, z3, z4, z5


class Decoder(fnn.Module):
    features: int = 64
    training: bool = True

    @fnn.compact
    def __call__(self, z1, z2, z3, z4, z5, t):
        z6_up = jax.image.resize(z5, shape=(z5.shape[0], z5.shape[1] * 2, z5.shape[2] * 2, z5.shape[3]),
                                 method='nearest')
        z6 = fnn.Conv(self.features * 8, kernel_size=(2, 2))(z6_up)
        z6 = fnn.relu(z6)
        z6 = jnp.concatenate([z4, z6], axis=3)
        z6 = fnn.Conv(self.features * 8, kernel_size=(3, 3))(z6)
        z6 = fnn.relu(z6)
        z6 = fnn.Conv(self.features * 8, kernel_size=(3, 3))(z6)
        #z6 = fnn.BatchNorm(use_running_average=not self.training)(z6)
        z6 = fnn.relu(z6)

        z7_up = jax.image.resize(z6, shape=(z6.shape[0], z6.shape[1] * 2, z6.shape[2] * 2, z6.shape[3]),
                                 method='nearest')
        z7 = fnn.Conv(self.features * 4, kernel_size=(2, 2))(z7_up)
        z7 = fnn.relu(z7)
        z7 = jnp.concatenate([z3, z7], axis=3)
        z7 = fnn.Conv(self.features * 4, kernel_size=(3, 3))(z7)
        z7 = fnn.relu(z7)
        z7 = fnn.Conv(self.features * 4, kernel_size=(3, 3))(z7)
        #z7 = fnn.BatchNorm(use_running_average=not self.training)(z7)
        z7 = fnn.relu(z7)

        z8_up = jax.image.resize(z7, shape=(z7.shape[0], z7.shape[1] * 2, z7.shape[2] * 2, z7.shape[3]),
                                 method='nearest')
        z8 = fnn.Conv(self.features * 2, kernel_size=(2, 2))(z8_up)
        z8_t = jnp.ones(z8.shape) * t
        z8 = fnn.relu(z8 + z8_t)
        z8 = jnp.concatenate([z2, z8], axis=3)
        z8 = fnn.Conv(self.features * 2, kernel_size=(3, 3))(z8)
        z8 = fnn.relu(z8)
        z8 = fnn.Conv(self.features * 2, kernel_size=(3, 3))(z8)
        #z8 = fnn.BatchNorm(use_running_average=not self.training)(z8)
        z8 = fnn.relu(z8)

        z9_up = jax.image.resize(z8, shape=(z8.shape[0], z8.shape[1] * 2, z8.shape[2] * 2, z8.shape[3]),
                                 method='nearest')
        z9 = fnn.Conv(self.features, kernel_size=(2, 2))(z9_up)
        z9_t = jnp.ones(z9.shape) * t
        z9 = fnn.relu(z9 + z9_t)
        z9 = jnp.concatenate([z1, z9], axis=3)
        z9 = fnn.Conv(self.features, kernel_size=(3, 3))(z9)
        z9 = fnn.relu(z9)
        z9 = fnn.Conv(self.features, kernel_size=(3, 3))(z9)
        #z9 = fnn.BatchNorm(use_running_average=not self.training)(z9)
        z9 = fnn.relu(z9)

        y = fnn.Conv(1, kernel_size=(1, 1))(z9)
        y = fnn.sigmoid(y)

        return y


class UNet(fnn.Module):
    features: int = 64
    training: bool = True

    @fnn.compact
    def __call__(self, x, t):
        z1, z2, z3, z4, z5 = Encoder(self.features, self.training)(x, t)
        y = Decoder(self.features, self.training)(z1, z2, z3, z4, z5, t)

        return y
