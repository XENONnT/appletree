import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm

class JAXCustomSampler:
    params = (
        1.1044, 0.36245, 6393.6,
        0.20664, 0.08045, 1.3597,
        33.003, 6.8804, 412.77,
        13.137, 33.066
    )

    def __init__(self, xrange=(0.0, 200.0), resolution=100_000):
        self.resolution = resolution
        self.x_grid = jnp.linspace(*xrange, self.resolution)
        self.pdf_grid = self._compute_pdf(self.x_grid)
        dx = self.x_grid[1] - self.x_grid[0]
        cdf = jnp.cumsum(self.pdf_grid) * dx
        self.cdf_grid = cdf / cdf[-1]

    def _low_area_part(self, x, mu, sigma, A, muT, sigmaT):
        threshold = norm.cdf(x, loc=muT, scale=sigmaT)
        return (
            A * norm.pdf(x, loc=mu, scale=sigma) *
            threshold *
            jnp.heaviside(-(x - 1.6), 1.0) *
            jnp.heaviside(x - 0.05, 1.0)
        )

    def _high_area_part(self, x, N0, tau, mu, sigma, A, z, tau2):
        return (
            N0 * jnp.exp(-x / tau) +
            z * jnp.exp(-x / tau2) +
            A * norm.pdf(x, loc=mu, scale=sigma)
        ) * jnp.heaviside(x - 1.6, 0.0)

    def _compute_pdf(self, x):
        (
            mu_s, sigma_s, A_s,
            muT, sigmaT,
            tau, mu, sigma,
            A, z, tau2
        ) = self.params

        N0_1p6 = self._low_area_part(1.6, mu_s, sigma_s, A_s, muT, sigmaT)
        N0 = N0_1p6 * jnp.exp(1.6 / tau) - z * jnp.exp(1.6 * (1 / tau - 1 / tau2))

        return (
            self._low_area_part(x, mu_s, sigma_s, A_s, muT, sigmaT) +
            self._high_area_part(x, N0, tau, mu, sigma, A, z, tau2)
        )

    def sample(self, key, n_samples):
        """Samples from the precomputed inverse CDF using JAX-native ops"""
        key, seed = random.split(key)
        u = random.uniform(seed, shape=(n_samples,), minval=0.0, maxval=1.0)
        return key, self._invert_cdf(u)

    def _invert_cdf(self, u):
        idx = jnp.searchsorted(self.cdf_grid, u, side="left") - 1
        idx = jnp.clip(idx, 0, self.resolution - 2)

        x0 = self.x_grid[idx]
        x1 = self.x_grid[idx + 1]
        y0 = self.cdf_grid[idx]
        y1 = self.cdf_grid[idx + 1]

        t = (u - y0) / (y1 - y0 + 1e-10)  # linear interpolation
        return x0 + t * (x1 - x0)