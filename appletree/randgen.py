import os
from warnings import warn
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

# Threshold for hybrid binomial: use exact Bernoulli sum for n <= threshold,
# Cornish-Fisher corrected normal approximation for n > threshold
HYBRID_BERNOULLI_THRESHOLD = 128

# Environment variable to enable hybrid binomial sampling
# Set APPLETREE_USE_HYBRID=1 to use Bernoulli sum (small n) + Cornish-Fisher (large n)
USE_HYBRID = os.environ.get("APPLETREE_USE_HYBRID", "0") == "1"
if USE_HYBRID:
    print(f"Using HYBRID binomial sampling (threshold={HYBRID_BERNOULLI_THRESHOLD})")

if jax.config.x64_enabled:
    INT = np.int64
    FLOAT = np.float64
else:
    INT = np.int32
    FLOAT = np.float32


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
def gamma(key, alpha, beta, shape=()):
    """Gamma distribution random sampler.

    Args:
        key: seed for random generator.
        alpha: <jnp.array>-like shape in gamma distribution.
        beta: <jnp.array>-like rate in normal distribution.
        shape: output shape.
            If not given, output has shape jnp.broadcast_shapes(jnp.shape(alpha), jnp.shape(beta)).

    Returns:
        an updated seed, random variables.

    """
    key, seed = random.split(key)

    shape = shape or jnp.broadcast_shapes(jnp.shape(alpha), jnp.shape(beta))
    alpha = jnp.broadcast_to(alpha, shape).astype(FLOAT)
    beta = jnp.broadcast_to(beta, shape).astype(FLOAT)

    rvs = random.gamma(seed, alpha, shape=shape) / beta
    return key, rvs.astype(FLOAT)


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
def binomial_hybrid(key, p, n, shape=(), threshold=HYBRID_BERNOULLI_THRESHOLD):
    """Hybrid binomial random sampler: Bernoulli sum + Cornish-Fisher expansion.

    For small n (n <= threshold): uses exact Bernoulli sum (no approximation error).
    For large n (n > threshold): uses Cornish-Fisher corrected normal approximation.

    This hybrid approach is faster than exact binomial while maintaining exact accuracy
    for small n where the normal approximation is least accurate.

    Args:
        key: seed for random generator.
        p: <jnp.array>-like probability in binomial distribution.
        n: <jnp.array>-like count in binomial distribution.
        shape: output shape.
            If not given, output has shape jnp.broadcast_shapes(jnp.shape(p), jnp.shape(n)).
        threshold: use Bernoulli sum for n <= threshold, CF for n > threshold.

    Returns:
        an updated seed, random variables.

    """
    key, key_cf, key_bernoulli = random.split(key, 3)

    # Determine output shape
    shape = shape or jnp.broadcast_shapes(jnp.shape(n), jnp.shape(p))

    # Broadcast inputs to output shape
    p = jnp.broadcast_to(p, shape).astype(FLOAT)
    n = jnp.broadcast_to(n, shape).astype(FLOAT)

    # ===== Cornish-Fisher path (for n > threshold) =====
    # Symmetry reduction: use p0 = min(p, 1-p) to reduce skewness
    flip = p > 0.5
    p0 = jnp.where(flip, 1.0 - p, p)

    # Compute moments
    q0 = 1.0 - p0
    mu = n * p0
    var = n * p0 * q0
    sigma = jnp.sqrt(var + 1e-12)
    var_safe = var + 1e-12

    # Skewness and excess kurtosis
    gamma1 = (q0 - p0) / (sigma + 1e-12)
    gamma2 = (1.0 - 6.0 * p0 * q0) / var_safe

    # Sample standard normal
    z = random.normal(key_cf, shape=shape, dtype=FLOAT)

    # Second-order Cornish-Fisher expansion
    z2 = z * z
    z3 = z2 * z
    gamma1_sq = gamma1 * gamma1

    z_cf = (
        z
        + (gamma1 / 6.0) * (z2 - 1.0)
        + (gamma2 / 24.0) * (z3 - 3.0 * z)
        - (gamma1_sq / 36.0) * (2.0 * z3 - 5.0 * z)
    )

    # Transform to binomial scale
    x_cf = mu + sigma * z_cf
    x_cf = jnp.round(x_cf)
    x_cf = jnp.clip(x_cf, 0.0, n)

    # Flip back if we used symmetry reduction
    x_cf = jnp.where(flip, n - x_cf, x_cf)

    # ===== Bernoulli sum path (for n <= threshold, exact) =====
    # Optimized: transposed layout (N, threshold) for better GPU memory coalescing
    n_flat = n.ravel()
    p_flat = p.ravel()
    num_elements = n_flat.shape[0]

    # Generate uniform random numbers for Bernoulli trials
    uniforms = random.uniform(key_bernoulli, (num_elements, threshold), dtype=FLOAT)

    # Create mask for valid trials: only count trials where trial_idx < n
    trial_idx = jnp.arange(threshold, dtype=FLOAT)[None, :]
    mask = trial_idx < n_flat[:, None]

    # Bernoulli trials: uniform < p gives success
    successes = (uniforms < p_flat[:, None]) & mask
    x_bernoulli = jnp.sum(successes, axis=1).reshape(shape)

    # ===== Combine paths based on n =====
    x = jnp.where(n <= threshold, x_bernoulli, x_cf)

    # Handle edge cases exactly
    x = jnp.where(p <= 0.0, 0.0, x)
    x = jnp.where(p >= 1.0, n, x)
    x = jnp.where(n <= 0.0, 0.0, x)

    return key, x.astype(INT)


if USE_HYBRID:
    # Use hybrid binomial (Bernoulli sum for small n, Cornish-Fisher for large n)
    @export
    @partial(jit, static_argnums=(3,))
    def binomial(key, p, n, shape=()):
        """Binomial random sampler (hybrid approximation).

        Uses exact Bernoulli sum for small n (n <= threshold).
        Uses Cornish-Fisher corrected normal approximation for large n.

        Args:
            key: seed for random generator.
            p: <jnp.array>-like probability in binomial distribution.
            n: <jnp.array>-like count in binomial distribution.
            shape: output shape.
                If not given, output has shape jnp.broadcast_shapes(jnp.shape(p), jnp.shape(n)).

        Returns:
            an updated seed, random variables.

        """
        return binomial_hybrid(key, p, n, shape)

elif hasattr(random, "binomial"):

    @export
    @partial(jit, static_argnums=(3,))
    def binomial(key, p, n, shape=()):
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

        key, seed = random.split(key)

        shape = shape or jnp.broadcast_shapes(jnp.shape(p), jnp.shape(n))
        p = jnp.broadcast_to(p, shape).astype(FLOAT)
        n = jnp.broadcast_to(n, shape).astype(INT)

        rvs = random.binomial(seed, n, p, shape=shape)
        return key, rvs.astype(INT)

else:
    warn("random.binomial is not available, using numpyro's _binomial_dispatch instead.")
    if os.environ.get("DO_NOT_USE_APPROX_IN_BINOM") is None:
        ALWAYS_USE_NORMAL_APPROX_IN_BINOM = True
        print("Using Normal as an approximation of Binomial")
    else:
        ALWAYS_USE_NORMAL_APPROX_IN_BINOM = False
        print("Using accurate Binomial, not Normal approximation")

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
@partial(jit, static_argnums=(3,))
def negative_binomial(key, p, n, shape=()):
    """Negative binomial distribution random sampler.

    Using Gamma–Poisson mixture.
        Args:
            key: seed for random generator.
            p: <jnp.array>-like probability of a single success in negative binomial distribution.
            n: <jnp.array>-like number of successes in negative binomial distribution.
            shape: output shape.
                If not given, output has shape jnp.broadcast_shapes(jnp.shape(p), jnp.shape(n)).

        Returns:
            an updated seed, random variables.

        References:
            1. https://en.wikipedia.org/wiki/Negative_binomial_distribution#Gamma%E2%80%93Poisson_mixture  # noqa
            2. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html

    """

    key, lam = gamma(key, n, p / (1 - p), shape)

    key, rvs = poisson(key, lam)
    return key, rvs


@export
@partial(jit, static_argnums=(3,))
def generalized_poisson(key, lam, eta, shape=()):
    """Generalized Poisson Distribution(GPD) random sampler.

    Args:
        key: seed for random generator.
        lam: <jnp.array>-like expectation(location parameter) in GPD.
        eta: <jnp.array>-like scale parameter in GPD, within [0, 1).
        shape: output shape. If not given, output has shape jnp.shape(lam).

    Returns:
        an updated seed, random variables.

    References:
        1. https://gist.github.com/danmackinlay/00e957b11c488539bd3e2a3804922b9d
        2. https://search.r-project.org/CRAN/refmans/LaplacesDemon/html/dist.Generalized.Poisson.html  # noqa

    """

    shape = shape or jnp.broadcast_shapes(jnp.shape(lam), jnp.shape(eta))
    lam = jnp.broadcast_to(lam, shape).astype(FLOAT)
    eta = jnp.broadcast_to(eta, shape).astype(FLOAT)

    key, population = poisson(key, lam * (1 - eta), shape)

    offspring = jnp.copy(population)

    def cond_fun(args):
        return jnp.any(args[1] > 0)

    def body_fun(args):
        key, offspring = poisson(args[0], eta * args[1])
        population = args[2] + offspring
        return key, offspring, population

    key, offspring, population = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (key, offspring, population),
    )

    return key, population.astype(INT)


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
