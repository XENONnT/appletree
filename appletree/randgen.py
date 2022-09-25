from functools import partial
from time import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import random, lax, jit, vmap

from appletree import utils
from appletree.utils import exporter

export, __all__ = exporter(export_self=False)

INT = np.int32
FLOAT = np.float32
ALWAYS_USE_NORMAL_APPROX_IN_BINOM = True


@export
def get_key(seed=None):
    if seed is None:
        seed = int(time())
    return random.PRNGKey(seed)


@export
@partial(jit, static_argnums=(3, ))
def uniform(key, vmin, vmax, shape=()):
    """
    Args
        key: seed for random generator
        vmin: <jnp.array>-like min in uniform distribution
        vmax: <jnp.array>-like max in uniform distribution
        shape: output shape. If not given, output has shape jnp.broadcast_shapes(jnp.shape(vmin), jnp.shape(vmax))
    Returns
        key: an updated seed
        rvs: random variables
    """
    key, seed = random.split(key)

    shape = shape or jnp.broadcast_shapes(jnp.shape(vmin), jnp.shape(vmax))
    vmin = jnp.broadcast_to(vmin, shape).astype(FLOAT)
    vmax = jnp.broadcast_to(vmax, shape).astype(FLOAT)
    
    rvs = random.uniform(key, shape, minval=vmin, maxval=vmax)
    return key, rvs.astype(FLOAT)


@export
@partial(jit, static_argnums=(2, ))
def poisson(key, lam, shape=()):
    """
    Args
        key: seed for random generator
        lam: <jnp.array>-like expectation in poisson distribution
        shape: output shape. If not given, output has shape jnp.shape(lam)
    Returns
        key: an updated seed
        rvs: random variables
    """
    key, seed = random.split(key)

    shape = shape or jnp.shape(lam)
    lam = jnp.broadcast_to(lam, shape).astype(FLOAT)

    rvs = random.poisson(seed, lam, shape=shape)
    return key, rvs.astype(INT)


@export
@partial(jit, static_argnums=(3, ))
def normal(key, mean, std, shape=()):
    """
    Args
        key: seed for random generator
        mean: <jnp.array>-like mean in normal distribution
        std: <jnp.array>-like std in normal distribution
        shape: output shape. If not given, output has shape jnp.broadcast_shapes(jnp.shape(mean), jnp.shape(std))
    Returns
        key: an updated seed
        rvs: random variables
    """
    key, seed = random.split(key)
    
    shape = shape or jnp.broadcast_shapes(jnp.shape(mean), jnp.shape(std))
    mean = jnp.broadcast_to(mean, shape).astype(FLOAT)
    std = jnp.broadcast_to(std, shape).astype(FLOAT)

    rvs = random.normal(seed, shape=shape) * std + mean
    return key, rvs.astype(FLOAT)


@export
@partial(jit, static_argnums=(5, ))
def truncate_normal(key, mean, std, vmin=None, vmax=None, shape=()):
    """
    Args
        key: seed for random generator
        mean: <jnp.array>-like mean in normal distribution
        std: <jnp.array>-like std in normal distribution
        vmin: <jnp.array>-like min value to clip. By default it's None. vmin and vmax cannot be both None.
        vmax: <jnp.array>-like max value to clip. By default it's None. vmin and vmax cannot be both None.
        shape: parameter passed to normal(..., shape=shape)
    Returns
        key: an updated seed
        rvs: random variables with shape jnp.broadcast_shapes(mean.shape, std.shape, vmin.shape, vmax.shape)
    """
    key, rvs = normal(key, mean, std, shape=shape)
    rvs = jnp.clip(rvs, a_min=vmin, a_max=vmax)
    return key, rvs.astype(FLOAT)


@export
@partial(jit, static_argnums=(3, 4))
def binomial(key, p, n, shape=(), always_use_normal=ALWAYS_USE_NORMAL_APPROX_IN_BINOM):
    """
    Args
        key: seed for random generator
        p: <jnp.array>-like probability in binomial distribution
        n: <jnp.array>-like count in binomial distribution
        shape: output shape. If not given, output has shape jnp.broadcast_shapes(jnp.shape(p), jnp.shape(n))
        always_use_normal: If true, then Norm(np, sqrt(npq)) is always used. 
                           Otherwise if np < 5, use the inversion method instead
    Returns
        key: an updated seed
        rvs: random variables
    """
    def _binomial_normal_approx_dispatch(seed, p, n):
        q = 1. - p
        mean = n * p
        std = jnp.sqrt(n * p * q)
        return jnp.clip(random.normal(seed) * std + mean, a_min=0.).round().astype(INT)

    def _binomial_dispatch(seed, p, n):
        use_normal_approx = (n * p >= 5.)
        return lax.cond(
            p*n >= 5.,
            (seed, p, n), lambda x: _binomial_normal_approx_dispatch(*x),
            (seed, p, n), lambda x: utils._binomial_dispatch(*x),
        )

    key, seed = random.split(key)

    shape = shape or lax.broadcast_shapes(jnp.shape(p), jnp.shape(n))
    p = jnp.reshape(jnp.broadcast_to(p, shape), -1).astype(FLOAT)
    n = jnp.reshape(jnp.broadcast_to(n, shape), -1).astype(INT)
    seed = random.split(seed, jnp.size(p))

    if always_use_normal:
        dispatch = _binomial_normal_approx_dispatch
    else:
        dispatch = _binomial_dispatch

    if jax.default_backend() == "cpu":
        ret = lax.map(lambda x: dispatch(*x), (seed, p, n))
    else:
        ret = vmap(lambda *x: dispatch(*x))(seed, p, n)
    return key, jnp.reshape(ret, shape).astype(INT)
