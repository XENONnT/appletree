from functools import partial
from jax import numpy as jnp
from jax import jit, vmap

from appletree.utils import exporter

export, __all__ = exporter(export_self=False)


@export
@jit
def make_hist_mesh_grid(sample, bins=10, weights=None):
    """Same as jnp.histogramdd."""
    hist, _ = jnp.histogramdd(sample, bins=bins, weights=weights)
    return hist


@export
@partial(jit, static_argnames=['method'])
def make_hist_irreg_bin_1d(sample, bins, weights, method='compare_all'):
    """Make a histogram with irregular binning.

    Args:
        sample: array with shape N.
        bins: array with shape M.
        weights: array with shape (N,).
        method: str passed to `jnp.searchsorted`. Can be 'scan', 'scan_unrolled', 
                'sort', or 'compare_all'. See jax documentation for details.

    """

    ind = jnp.searchsorted(bins, sample, method=method)

    hist = jnp.zeros(len(bins) + 1)
    hist = hist.at[ind].add(weights)

    return hist[1:-1]


@export
@partial(jit, static_argnames=['method'])
def make_hist_irreg_bin_2d(sample, bins_x, bins_y, weights, method='compare_all'):
    """Make a histogram with irregular binning.

    Args:
        sample: array with shape (N, 2).
        bins_x: array with shape (M1, ).
        bins_y: array with shape (M1-1, M2).
        weights: array with shape (N,).
        method: str passed to `jnp.searchsorted`. Can be 'scan', 'scan_unrolled', 
                'sort', or 'compare_all'. See jax documentation for details.

    """
    x = sample[:, 0]
    y = sample[:, 1]

    ind_x = jnp.searchsorted(bins_x, x, method=method)
    ind_y = vmap(lambda bins, data: jnp.searchsorted(bins, data, method=method), (0, 0), 0)(
        bins_y[ind_x - 1], y
    )

    bin_ind = jnp.stack((ind_x, ind_y))

    output_shape = (len(bins_x) + 1, bins_y.shape[-1] + 1)
    hist = jnp.zeros(output_shape)
    hist = hist.at[tuple(bin_ind)].add(weights)

    return hist[1:-1, 1:-1]
