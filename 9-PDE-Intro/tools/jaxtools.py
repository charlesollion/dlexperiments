import jax


def add_gradients(grad1, grad2, alpha=1.0, beta=1.0):
    """
    Add two gradient trees with weights
    """
    return jax.tree_multimap(lambda x, y: x*alpha+y*beta, grad1, grad2)

add_grads = jit(add_gradients)
