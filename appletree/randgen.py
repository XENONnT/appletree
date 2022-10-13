from functools import partial
from time import time

import jax
from jax import numpy as jnp
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
    """Generate a key for jax.random."""
    if seed is None:
        seed = int(time()*1e6)
    return random.PRNGKey(seed)


@export
@partial(jit, static_argnums=(3, ))
def uniform(key, vmin, vmax, shape=()):
    """Uniform random sampler.
    :param key: seed for random generator.
    :param vmin: <jnp.array>-like min in uniform distribution.
    :param vmax: <jnp.array>-like max in uniform distribution.
    :param shape: output shape. If not given, output has shape
    jnp.broadcast_shapes(jnp.shape(vmin), jnp.shape(vmax)).
    :return: an updated seed, random variables.
    """
    key, seed = random.split(key)

    shape = shape or jnp.broadcast_shapes(jnp.shape(vmin), jnp.shape(vmax))
    vmin = jnp.broadcast_to(vmin, shape).astype(FLOAT)
    vmax = jnp.broadcast_to(vmax, shape).astype(FLOAT)

    rvs = random.uniform(seed, shape, minval=vmin, maxval=vmax)
    return key, rvs.astype(FLOAT)


@export
@partial(jit, static_argnums=(2, ))
def poisson(key, lam, shape=()):
    """Poisson random sampler.
    :param key: seed for random generator.
    :param lam: <jnp.array>-like expectation in poisson distribution.
    :param shape: output shape. If not given, output has shape jnp.shape(lam).
    :return: an updated seed, random variables.
    """
    key, seed = random.split(key)

    shape = shape or jnp.shape(lam)
    lam = jnp.broadcast_to(lam, shape).astype(FLOAT)

    rvs = random.poisson(seed, lam, shape=shape)
    return key, rvs.astype(INT)


@export
@partial(jit, static_argnums=(3, ))
def normal(key, mean, std, shape=()):
    """Normal distribution random sampler.
    :param key: seed for random generator.
    :param mean: <jnp.array>-like mean in normal distribution.
    :param std: <jnp.array>-like std in normal distribution.
    :param shape: output shape. If not given, output has shape
    jnp.broadcast_shapes(jnp.shape(mean), jnp.shape(std)).
    :return: an updated seed, random variables.
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
    """Truncated normal distribution random sampler.
    :param key: seed for random generator.
    :param mean: <jnp.array>-like mean in normal distribution.
    :param std: <jnp.array>-like std in normal distribution.
    :param vmin: <jnp.array>-like min value to clip.
    By default it's None. vmin and vmax cannot be both None.
    :param vmax: <jnp.array>-like max value to clip.
    By default it's None. vmin and vmax cannot be both None.
    :param shape: parameter passed to normal(..., shape=shape)
    :return: an updated seed, random variables.
    """
    key, rvs = normal(key, mean, std, shape=shape)
    rvs = jnp.clip(rvs, a_min=vmin, a_max=vmax)
    return key, rvs.astype(FLOAT)


@export
@partial(jit, static_argnums=(3, 4))
def binomial(key, p, n, shape=(), always_use_normal=ALWAYS_USE_NORMAL_APPROX_IN_BINOM):
    """Binomial random sampler.
    :param key: seed for random generator.
    :param p: <jnp.array>-like probability in binomial distribution.
    :param n: <jnp.array>-like count in binomial distribution.
    :param shape: output shape. If not given, output has shape
    jnp.broadcast_shapes(jnp.shape(p), jnp.shape(n)).
    :param always_use_normal: If true, then Norm(np, sqrt(npq)) is always used.
    Otherwise if np < 5, use the inversion method instead.
    :return: an updated seed, random variables.
    """
    def _binomial_normal_approx_dispatch(seed, p, n):
        q = 1. - p
        mean = n * p
        std = jnp.sqrt(n * p * q)
        rvs = jnp.clip(random.normal(seed) * std + mean, a_min=0.)
        return rvs.round().astype(INT)

    def _binomial_dispatch(seed, p, n):
        use_normal_approx = (n * p >= 5.)
        return lax.cond(
            use_normal_approx,
            (seed, p, n), lambda x: _binomial_normal_approx_dispatch(*x),
            (seed, p, n), lambda x: utils._binomial_dispatch_numpyro(*x),
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


@export
@jit
def uniform_key_vectorized(key):
    """Uniform(0,1) distribution sampler, vectorized by key.
    Note: key won't be updated!
    :param key: seed for random generator, with shape (N, 2)
    :return: random varibles with shape (N, )
    """
    sampler = vmap(jax.random.uniform, (0, ), 0)
    return sampler(key)
