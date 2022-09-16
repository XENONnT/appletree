import jax.numpy as jnp
from functools import partial
from jax import jit, vmap

@partial(jit, static_argnums=(1,))
def make_hist_mesh_grid(sample, bins=10, weights=None):
    return jnp.histogramdd(sample, bins=bins, weights=weights)

@jit
def make_hist_irreg_bin_2d(sample, bins_x, bins_y, weights=None):
    """
    sample : (N, 2)
    bins_x : (M1, )
    bins_y : (M1-1, M2)
    weights : (N, )
    
    return : (M1-1, M2-1)
    """
    weights = jnp.ones(len(sample))

    x = sample[:, 0]
    y = sample[:, 1]

    ind_x = jnp.searchsorted(bins_x, x)
    ind_y = vmap(jnp.searchsorted, (0, 0), 0)(bins_y[ind_x-1], y)

    bin_ind = jnp.stack((ind_x, ind_y))

    hist = jnp.zeros((len(bins_x)+1, bins_y.shape[-1]+1))
    hist = hist.at[tuple(bin_ind)].add(weights)
    
    return hist[1:-1, 1:-1]