import os
import re
from time import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import GOFevaluation

def exporter(export_self=False):
    """
    Export utility modified from https://stackoverflow.com/a/41895194
    Returns export decorator, __all__ list
    """
    all_ = []
    if export_self:
        all_.append('exporter')

    def decorator(obj):
        all_.append(obj.__name__)
        return obj

    return decorator, all_

export, __all__ = exporter(export_self=True)

@export
def camel_to_snake(x):
    """Convert x from CamelCase to snake_case"""
    # From https://stackoverflow.com/questions/1175208
    x = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', x)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', x).lower()

@export
def timeit(indent=""):
    """
    Use timeit as a decorator.
    """
    def _timeit(func, indent):
        name = func.__name__
        def _func(*args, **kwargs):
            print(indent + ' Function <%s> starts. '%name)
            start = time()
            res = func(*args, **kwargs)
            print(indent + ' Function <%s> ends! Time cost = %f msec. '%(name, (time()-start)*1e3))
            return res
        return _func
    if isinstance(indent, str):
        return lambda func: _timeit(func, indent)
    else:
        return _timeit(indent, "")

    
@export
def set_gpu_memory_usage(fraction=0.3):
    if fraction > 1.:
        fraction = 1
    if fraction <= 0:
        raise ValueError("fraction must be positive!")
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{fraction:.2f}"
    
    
@export
def get_equiprob_bins_2d(data, n_partitions, order=[0,1], x_clip=[-np.inf, +np.inf], y_clip=[-np.inf, +np.inf], which_np=np):
    mask = (data[:, 0] > x_clip[0]) & (data[:, 0] < x_clip[1])
    mask &= (data[:, 1] > y_clip[0]) & (data[:, 1] < y_clip[1])
    
    x_bins, y_bins = GOFevaluation.utils._get_equiprobable_binning(data[mask], n_partitions, order=order)
    x_bins = np.clip(x_bins, *x_clip)
    y_bins = np.clip(y_bins, *y_clip)
    
    return which_np.array(x_bins), which_np.array(y_bins)


@export
def plot_irreg_histogram_2d(bins_x, bins_y, hist, **kwargs):
    hist = np.asarray(hist)
    bins_x = np.asarray(bins_x)
    bins_y = np.asarray(bins_y)
    
    density = kwargs.get('density', False)
    cmap = mpl.cm.get_cmap("RdBu_r")

    loc = []
    width = []
    height = []
    area = []
    n = []
    
    for i in range(len(hist)):
        for j in range(len(hist[i])):
            x_lower = bins_x[i]
            x_upper = bins_x[i+1]
            y_lower = bins_y[i, j]
            y_upper = bins_y[i, j+1]

            loc.append((x_lower, y_lower))
            width.append(x_upper - x_lower)
            height.append(y_upper - y_lower)
            area.append((x_upper - x_lower) * (y_upper - y_lower))
            n.append(hist[i, j])

    loc = np.asarray(loc)
    width = np.asarray(width)
    height = np.asarray(height)
    area = np.asarray(area)
    n = np.asarray(n)

    if density:
        norm = mpl.colors.Normalize(
            vmin=kwargs.get('vmin', np.min(n/area)),
            vmax=kwargs.get('vmax', np.max(n/area)),
            clip=False
        )
    else:
        norm = mpl.colors.Normalize(
            vmin=kwargs.get('vmin', np.min(n)),
            vmax=kwargs.get('vmax', np.max(n)),
            clip=False
        )

    ax = plt.subplot()
    for i in range(len(loc)):
        rec = Rectangle(
            loc[i],
            width[i],
            height[i],
            facecolor=cmap(norm(n[i]/area[i] if density else n[i])),
            edgecolor='k'
        )
        ax.add_patch(rec)
        
    fig = plt.gcf() 
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        label=('# events / bin size' if density else '# events')
    )

    ax.set_xlim(np.min(bins_x), np.max(bins_x))
    ax.set_ylim(np.min(bins_y), np.max(bins_y))

    return ax


# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from functools import partial, update_wrapper
import math

import numpy as np

import jax
from jax import jit, lax, random, vmap
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

# Parameters for Transformed Rejection with Squeeze (TRS) algorithm - page 3.
_tr_params = namedtuple(
    "tr_params", ["c", "b", "a", "alpha", "u_r", "v_r", "m", "log_p", "log1_p", "log_h"]
)


def _get_tr_params(n, p):
    # See Table 1. Additionally, we pre-compute log(p), log1(-p) and the
    # constant terms, that depend only on (n, p, m) in log(f(k)) (bottom of page 5).
    mu = n * p
    spq = jnp.sqrt(mu * (1 - p))
    c = mu + 0.5
    b = 1.15 + 2.53 * spq
    a = -0.0873 + 0.0248 * b + 0.01 * p
    alpha = (2.83 + 5.1 / b) * spq
    u_r = 0.43
    v_r = 0.92 - 4.2 / b
    m = jnp.floor((n + 1) * p).astype(n.dtype)
    log_p = jnp.log(p)
    log1_p = jnp.log1p(-p)
    log_h = (m + 0.5) * (jnp.log((m + 1.0) / (n - m + 1.0)) + log1_p - log_p) + (
        stirling_approx_tail(m) + stirling_approx_tail(n - m)
    )
    return _tr_params(c, b, a, alpha, u_r, v_r, m, log_p, log1_p, log_h)


def stirling_approx_tail(k):
    precomputed = jnp.array(
        [
            0.08106146679532726,
            0.04134069595540929,
            0.02767792568499834,
            0.02079067210376509,
            0.01664469118982119,
            0.01387612882307075,
            0.01189670994589177,
            0.01041126526197209,
            0.009255462182712733,
            0.008330563433362871,
        ]
    )
    kp1 = k + 1
    kp1sq = (k + 1) ** 2
    return jnp.where(
        k < 10,
        precomputed[k],
        (1.0 / 12 - (1.0 / 360 - (1.0 / 1260) / kp1sq) / kp1sq) / kp1,
    )


_binomial_mu_thresh = 10


def _binomial_btrs(key, p, n):
    """
    Based on the transformed rejection sampling algorithm (BTRS) from the
    following reference:

    Hormann, "The Generation of Binonmial Random Variates"
    (https://core.ac.uk/download/pdf/11007254.pdf)
    """

    def _btrs_body_fn(val):
        _, key, _, _ = val
        key, key_u, key_v = random.split(key, 3)
        u = random.uniform(key_u)
        v = random.uniform(key_v)
        u = u - 0.5
        k = jnp.floor(
            (2 * tr_params.a / (0.5 - jnp.abs(u)) + tr_params.b) * u + tr_params.c
        ).astype(n.dtype)
        return k, key, u, v

    def _btrs_cond_fn(val):
        def accept_fn(k, u, v):
            # See acceptance condition in Step 3. (Page 3) of TRS algorithm
            # v <= f(k) * g_grad(u) / alpha

            m = tr_params.m
            log_p = tr_params.log_p
            log1_p = tr_params.log1_p
            # See: formula for log(f(k)) at bottom of Page 5.
            log_f = (
                (n + 1.0) * jnp.log((n - m + 1.0) / (n - k + 1.0))
                + (k + 0.5) * (jnp.log((n - k + 1.0) / (k + 1.0)) + log_p - log1_p)
                + (stirling_approx_tail(k) - stirling_approx_tail(n - k))
                + tr_params.log_h
            )
            g = (tr_params.a / (0.5 - jnp.abs(u)) ** 2) + tr_params.b
            return jnp.log((v * tr_params.alpha) / g) <= log_f

        k, key, u, v = val
        early_accept = (jnp.abs(u) <= tr_params.u_r) & (v <= tr_params.v_r)
        early_reject = (k < 0) | (k > n)
        # when vmapped _binomial_dispatch will convert the cond condition into
        # a HLO select that will execute both branches. This is a workaround
        # that avoids the resulting infinite loop when p=0. This should also
        # improve performance in less catastrophic cases.
        cond_exclude_small_mu = p * n >= _binomial_mu_thresh
        cond_main = lax.cond(
            early_accept | early_reject,
            (),
            lambda _: ~early_accept,
            (k, u, v),
            lambda x: ~accept_fn(*x),
        )
        return cond_exclude_small_mu & cond_main

    tr_params = _get_tr_params(n, p)
    ret = lax.while_loop(
        _btrs_cond_fn, _btrs_body_fn, (-1, key, 1.0, 1.0)
    )  # use k=-1 initially so that cond_fn returns True
    return ret[0]


def _binomial_inversion(key, p, n):
    def _binom_inv_body_fn(val):
        i, key, geom_acc = val
        key, key_u = random.split(key)
        u = random.uniform(key_u)
        geom = jnp.floor(jnp.log1p(-u) / log1_p) + 1
        geom_acc = geom_acc + geom
        return i + 1, key, geom_acc

    def _binom_inv_cond_fn(val):
        i, _, geom_acc = val
        # see the note on cond_exclude_small_mu in _binomial_btrs
        # this cond_exclude_large_mu is unnecessary for correctness but will
        # still improve performance.
        cond_exclude_large_mu = p * n < _binomial_mu_thresh
        return cond_exclude_large_mu & (geom_acc <= n)

    log1_p = jnp.log1p(-p)
    ret = lax.while_loop(_binom_inv_cond_fn, _binom_inv_body_fn, (-1, key, 0.0))
    return ret[0]


def _binomial_dispatch(key, p, n):
    def dispatch(key, p, n):
        is_le_mid = p <= 0.5
        pq = jnp.where(is_le_mid, p, 1 - p)
        mu = n * pq
        k = lax.cond(
            mu < _binomial_mu_thresh,
            (key, pq, n),
            lambda x: _binomial_inversion(*x),
            (key, pq, n),
            lambda x: _binomial_btrs(*x),
        )
        return jnp.where(is_le_mid, k, n - k)

    # Return 0 for nan `p` or negative `n`, since nan values are not allowed for integer types
    cond0 = jnp.isfinite(p) & (n > 0) & (p > 0)
    return lax.cond(
        cond0 & (p < 1),
        (key, p, n),
        lambda x: dispatch(*x),
        (),
        lambda _: jnp.where(cond0, n, 0),
    )


@partial(jit, static_argnums=(3,))
def _binomial(key, p, n, shape):
    shape = shape or lax.broadcast_shapes(jnp.shape(p), jnp.shape(n))
    # reshape to map over axis 0
    p = jnp.reshape(jnp.broadcast_to(p, shape), -1)
    n = jnp.reshape(jnp.broadcast_to(n, shape), -1)
    key = random.split(key, jnp.size(p))
    if jax.default_backend() == "cpu":
        ret = lax.map(lambda x: _binomial_dispatch(*x), (key, p, n))
    else:
        ret = vmap(lambda *x: _binomial_dispatch(*x))(key, p, n)
    return jnp.reshape(ret, shape)


@export
def binomial(key, p, n=1, shape=()):
    return _binomial(key, p, n, shape)
