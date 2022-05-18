import jax
import jax.numpy as np
from jax import jit

kernel_simple = np.array([[0.,  1., 0.],
                          [1., -4., 1.],
                          [0.,  1., 0.]])
kernel_large = np.array([[1.,  1., 1.],
                          [1., -8., 1.],
                          [1.,  1., 1.]])

def laplacian_grid(grid, ker=kernel_simple, pad=1):
    return jax.scipy.signal.convolve2d(grid, ker)[pad:-pad, pad:-pad]


def add_gradients(grad1, grad2, alpha=1.0, beta=1.0):
    """
    Add two gradient trees with weights
    """
    return jax.tree_multimap(lambda x, y: x*alpha+y*beta, grad1, grad2)

add_grads = jit(add_gradients)
