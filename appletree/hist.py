from jax import numpy as jnp
from jax import jit, vmap

from appletree.utils import exporter

export, __all__ = exporter(export_self=False)


@export
def make_hist_mesh_grid(sample, bins=10, weights=None):
    """Same as jnp.histogramdd."""
    hist, _ = jnp.histogramdd(sample, bins=bins, weights=weights)
    return hist


@export
@jit
def make_hist_irreg_bin_2d(sample, bins_x, bins_y, weights):
    """Make a histogram with irregular binning.
    :param sample: array with shape (N, 2)
    :param bins_x: array with shape (M1, )
    :param bins_y: array with shape (M1-1, M2)
    :param weights: array with shape (N, )
    """
    x = sample[:, 0]
    y = sample[:, 1]

    ind_x = jnp.searchsorted(bins_x, x)
    ind_y = vmap(jnp.searchsorted, (0, 0), 0)(bins_y[ind_x-1], y)

    bin_ind = jnp.stack((ind_x, ind_y))

    output_shape = (len(bins_x)+1, bins_y.shape[-1]+1)
    hist = jnp.zeros(output_shape)
    hist = hist.at[tuple(bin_ind)].add(weights)

    return hist[1:-1, 1:-1]
