import os
from functools import partial
from time import time

import jax
from jax import numpy as jnp
import numpy as np
from jax import random, lax, jit, vmap
from numpyro.distributions.util import _binomial_dispatch as _binomial_dispatch_numpyro
from scipy.interpolate import interp1d

from appletree.utils import exporter

export, __all__ = exporter(export_self=False)

INT = np.int32
FLOAT = np.float32

if os.environ.get("DO_NOT_USE_APPROX_IN_BINOM") is None:
    ALWAYS_USE_NORMAL_APPROX_IN_BINOM = True
    print("Using Normal as an approximation of Binomial")
else:
    ALWAYS_USE_NORMAL_APPROX_IN_BINOM = False
    print("Using accurate Binomial, not Normal approximation")


@export
def get_key(seed=None):
    """Generate a key for jax.random."""
    if seed is None:
        seed = int(time() * 1e6)
    return random.PRNGKey(seed)


@export
@partial(jit, static_argnums=(3,))
def uniform(key, vmin, vmax, shape=()):
    """Uniform random sampler.

    Args:
        key: seed for random generator.
        vmin: <jnp.array>-like min in uniform distribution.
        vmax: <jnp.array>-like max in uniform distribution.
        shape: output shape.
            If not given, output has shape jnp.broadcast_shapes(jnp.shape(vmin), jnp.shape(vmax)).

    Returns:
        an updated seed, random variables.

    """
    key, seed = random.split(key)

    shape = shape or jnp.broadcast_shapes(jnp.shape(vmin), jnp.shape(vmax))
    vmin = jnp.broadcast_to(vmin, shape).astype(FLOAT)
    vmax = jnp.broadcast_to(vmax, shape).astype(FLOAT)

    rvs = random.uniform(seed, shape, minval=vmin, maxval=vmax)
    return key, rvs.astype(FLOAT)


@export
@partial(jit, static_argnums=(2,))
def poisson(key, lam, shape=()):
    """Poisson random sampler.

    Args:
        key: seed for random generator.
        lam: <jnp.array>-like expectation in poisson distribution.
        shape: output shape. If not given, output has shape jnp.shape(lam).

    Returns:
        an updated seed, random variables.

    """
    key, seed = random.split(key)

    shape = shape or jnp.shape(lam)
    lam = jnp.broadcast_to(lam, shape).astype(FLOAT)

    rvs = random.poisson(seed, lam, shape=shape)
    return key, rvs.astype(INT)


@export
@partial(jit, static_argnums=(3,))
def normal(key, mean, std, shape=()):
    """Normal distribution random sampler.

    Args:
        key: seed for random generator.
        mean: <jnp.array>-like mean in normal distribution.
        std: <jnp.array>-like std in normal distribution.
        shape: output shape.
            If not given, output has shape jnp.broadcast_shapes(jnp.shape(mean), jnp.shape(std)).

    Returns:
        an updated seed, random variables.

    """
    key, seed = random.split(key)

    shape = shape or jnp.broadcast_shapes(jnp.shape(mean), jnp.shape(std))
    mean = jnp.broadcast_to(mean, shape).astype(FLOAT)
    std = jnp.broadcast_to(std, shape).astype(FLOAT)

    rvs = random.normal(seed, shape=shape) * std + mean
    return key, rvs.astype(FLOAT)


@export
@partial(jit, static_argnums=(5,))
def truncate_normal(key, mean, std, vmin=None, vmax=None, shape=()):
    """Truncated normal distribution random sampler.

    Args:
        key: seed for random generator.
        mean: <jnp.array>-like mean in normal distribution.
        std: <jnp.array>-like std in normal distribution.
        vmin: <jnp.array>-like min value to clip. By default it's None.
            vmin and vmax cannot be both None.
        vmax: <jnp.array>-like max value to clip. By default it's None.
            vmin and vmax cannot be both None.
        shape: parameter passed to normal(..., shape=shape)

    Returns:
        an updated seed, random variables.

    """
    key, rvs = normal(key, mean, std, shape=shape)
    rvs = jnp.clip(rvs, a_min=vmin, a_max=vmax)
    return key, rvs.astype(FLOAT)


@export
@partial(jit, static_argnums=(4,))
def skewnormal(key, a, loc, scale, shape=()):
    """Skew-normal distribution random sampler.

    Args:
        key: seed for random generator.
        a: <jnp.array>-like skewness in skewnormal distribution.
        loc: <jnp.array>-like loc in skewnormal distribution.
        scale: <jnp.array>-like scale in skewnormal distribution.
        shape: output shape.
            If not given, output has shape jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale)).

    Returns:
        an updated seed, random variables.

    References:
        1. Dariush Ghorbanzadeh, Luan Jaupi, Philippe Durand.
        A Method to Simulate the Skew Normal Distribution.
        Applied Mathematics, 2014, 5 (13), pp.2073-2076. ff10.4236/am.2014.513201ff. ffhal02467997

    """
    shape = shape or jnp.broadcast_shapes(jnp.shape(a), jnp.shape(loc), jnp.shape(scale))
    a = jnp.broadcast_to(a, shape).astype(FLOAT)
    loc = jnp.broadcast_to(loc, shape).astype(FLOAT)
    scale = jnp.broadcast_to(scale, shape).astype(FLOAT)

    key, seed = random.split(key)
    rvs0 = random.normal(seed, shape=shape)
    key, seed = random.split(key)
    rvs1 = random.normal(seed, shape=shape)
    rvs = (a * jnp.abs(rvs0) + rvs1) / jnp.sqrt(1 + a**2)
    rvs = rvs * scale + loc
    return key, rvs.astype(FLOAT)


@export
@partial(jit, static_argnums=(2,))
def bernoulli(key, p, shape=()):
    """Bernoulli random sampler.

    Args:
        key: seed for random generator.
        p: <jnp.array>-like probability in bernoulli distribution.
        shape: output shape. If not given, output has shape jnp.shape(lam).

    Returns:
        an updated seed, random variables.

    """
    key, seed = random.split(key)

    shape = shape or jnp.shape(p)
    p = jnp.broadcast_to(p, shape).astype(FLOAT)

    rvs = random.bernoulli(seed, p, shape=shape)
    return key, rvs.astype(INT)


@export
@partial(jit, static_argnums=(3, 4))
def binomial(key, p, n, shape=(), always_use_normal=ALWAYS_USE_NORMAL_APPROX_IN_BINOM):
    """Binomial random sampler.

    Args:
        key: seed for random generator.
        p: <jnp.array>-like probability in binomial distribution.
        n: <jnp.array>-like count in binomial distribution.
        shape: output shape.
            If not given, output has shape jnp.broadcast_shapes(jnp.shape(p), jnp.shape(n)).
        always_use_normal: If true, then Norm(np, sqrt(npq)) is always used.
            Otherwise if n * p < 5, use the inversion method instead.

    Returns:
        an updated seed, random variables.

    """

    def _binomial_normal_approx_dispatch(seed, p, n):
        q = 1.0 - p
        mean = n * p
        std = jnp.sqrt(n * p * q)
        rvs = jnp.clip(random.normal(seed) * std + mean, a_min=0.0, a_max=n)
        return rvs.round().astype(INT)

    def _binomial_dispatch(seed, p, n):
        use_normal_approx = n * p >= 5.0
        return lax.cond(
            use_normal_approx,
            (seed, p, n),
            lambda x: _binomial_normal_approx_dispatch(*x),
            (seed, p, n),
            lambda x: _binomial_dispatch_numpyro(*x),
        )

    key, seed = random.split(key)

    shape = shape or lax.broadcast_shapes(jnp.shape(p), jnp.shape(n))
    p = jnp.reshape(jnp.broadcast_to(p, shape), -1)
    n = jnp.reshape(jnp.broadcast_to(n, shape), -1)
    seed = random.split(seed, jnp.size(p))

    if always_use_normal:
        dispatch = _binomial_normal_approx_dispatch
    else:
        dispatch = _binomial_dispatch

    if jax.default_backend() == "cpu":
        ret = lax.map(lambda x: dispatch(*x), (seed, p, n))
    else:
        ret = vmap(lambda *x: dispatch(*x))(seed, p, n)
    return key, jnp.reshape(ret, shape)


@export
@jit
def uniform_key_vectorized(key):
    """Uniform(0,1) distribution sampler, vectorized by key.
    Note: key won't be updated!

    Args:
        key: seed for random generator, with shape (N, 2)

    Returns:
        random varibles with shape (N,)
    """
    sampler = vmap(jax.random.uniform, (0,), 0)
    return sampler(key)


class TwoHalfNorm:
    """Continuous distribution, two half Normal."""

    @staticmethod
    def rvs(mu=0, sigma_pos=1, sigma_neg=1, size=None):
        """Get random variables.

        Args:
            mu: float, 'center' value of the distribution.
            sigma_pos: float, Standard deviation of the distribution when variable larger than mu.
                Must be non-negative.
            sigma_neg: float, Standard deviation of the distribution when variable smaller than mu.
                Must be non-negative.
            size: int or tuple of ints, Output shape.

        Returns:
            random samples.

        """
        assert (sigma_pos > 0) and (sigma_neg > 0), "sigma should be positive"
        pos_half_prob = sigma_pos / (sigma_pos + sigma_neg)

        use_pos_half = np.random.uniform(size=size) < pos_half_prob
        use_neg_half = ~use_pos_half

        n_sigma = np.abs(np.random.normal(size=size))
        offset = use_pos_half * n_sigma * sigma_pos - use_neg_half * n_sigma * sigma_neg

        return offset + mu

    @staticmethod
    def logpdf(x, mu=0, sigma_pos=1, sigma_neg=1):
        """Log of the probability density function.

        Args:
            x: array, input variables.
            mu: float, 'center' value of the distribution.
            sigma_pos: float, Standard deviation of the distribution when variable larger than mu.
                Must be non-negative.
            sigma_neg: float, Standard deviation of the distribution when variable smaller than mu.
                Must be non-negative.
            size: int or tuple of ints, Output shape.

        Returns:
            log probability density function.

        """
        assert np.all(sigma_pos > 0) and np.all(sigma_neg > 0), "sigma should be positive"
        norm = 2 / (sigma_pos + sigma_neg) / np.sqrt(2 * np.pi)
        logpdf = np.where(
            x < mu, -((x - mu) ** 2) / sigma_neg**2 / 2, -((x - mu) ** 2) / sigma_pos**2 / 2
        )
        logpdf += np.log(norm)
        return logpdf


class BandTwoHalfNorm:
    """This is a TwoHalfNorm which quantifies uncertainty in y-axis, but we need to interpolate from
    x-axis."""

    def __init__(self, x, y, yerr_upper, yerr_lower):
        self.x = x
        self.y = interp1d(x, y, bounds_error=False, fill_value=np.nan)
        self.yerr_upper = interp1d(x, yerr_upper, bounds_error=False, fill_value=np.nan)
        self.yerr_lower = interp1d(x, yerr_lower, bounds_error=False, fill_value=np.nan)

    def logpdf(self, x, y):
        """We calculate along LLH where y-axis is random variable.

        We need to interpolate along x-axis to get y-axis's arguments.

        """
        mu = self.y(x)
        sigma_pos = self.yerr_upper(x)
        sigma_neg = self.yerr_lower(x)
        _mu = np.where(np.isnan(mu), 0, mu)
        _sigma_pos = np.where(np.isnan(sigma_pos), 1, sigma_pos)
        _sigma_neg = np.where(np.isnan(sigma_neg), 1, sigma_neg)
        logpdf = TwoHalfNorm.logpdf(x=y, mu=_mu, sigma_pos=_sigma_pos, sigma_neg=_sigma_neg)
        # If x out of range of self.x, return -np.inf
        logpdf = np.where(np.isnan(mu), -np.inf, logpdf)
        return logpdf


@export
@partial(jit, static_argnums=(4,))
def twohalfnorm(key, mu, sigma_pos, sigma_neg, shape=()):
    """JAX version of TwoHalfNorm.rvs.

    Args:
        key: seed for random generator.
        shape: output shape.
            If not given, output has shape jnp.broadcast_shapes(jnp.shape(mean), jnp.shape(std)).

    Returns:
        an updated seed, random variables.

    """
    key, seed = random.split(key)

    shape = shape or jnp.broadcast_shapes(jnp.shape(mu), jnp.shape(sigma_pos), jnp.shape(sigma_neg))
    mu = jnp.broadcast_to(mu, shape).astype(FLOAT)
    sigma_pos = jnp.broadcast_to(sigma_pos, shape).astype(FLOAT)
    sigma_neg = jnp.broadcast_to(sigma_neg, shape).astype(FLOAT)

    pos_half_prob = sigma_pos / (sigma_pos + sigma_neg)
    use_pos_half = random.uniform(seed, shape, minval=0.0, maxval=1.0) < pos_half_prob
    use_neg_half = ~use_pos_half

    n_sigma = jnp.abs(random.normal(seed, shape=shape))
    offset = use_pos_half * n_sigma * sigma_pos - use_neg_half * n_sigma * sigma_neg
    rvs = offset + mu

    return key, rvs.astype(FLOAT)
