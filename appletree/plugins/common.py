from functools import partial

import jax.numpy as jnp
from jax import jit

from appletree import randgen
from appletree.plugin import Plugin
from appletree.utils import exporter

export, __all__ = exporter()


@export
class EnergySpectra(Plugin):
    depends_on = ['batch_size']
    provides = ['energy']

    def __init__(self, lower=0.01, upper=20.):
        super().__init__()

        self.lower = lower
        self.upper = upper

    @partial(jit, static_argnums=(0, 2))
    def simulate(self, key, parameters, batch_size):
        key, energy = randgen.uniform(key, self.lower, self.upper, shape=(batch_size, ))
        return key, energy


@export
class PositionSpectra(Plugin):
    depends_on = ['batch_size']
    provides = ['x', 'y', 'z']

    def __init__(self, z_sim_min=-133.97, z_sim_max=-13.35, r_sim_max=60.):
        super().__init__()

        self.z_lower = z_sim_min
        self.z_upper = z_sim_max
        self.r_upper = r_sim_max

    @partial(jit, static_argnums=(0, 2))
    def simulate(self, key, parameters, batch_size):
        key, z = randgen.uniform(key, self.z_lower, self.z_upper, shape=(batch_size, ))
        key, r2 = randgen.uniform(key, 0, self.r_upper**2, shape=(batch_size, ))
        key, theta = randgen.uniform(key, 0, 2*jnp.pi, shape=(batch_size, ))

        r = jnp.sqrt(r2)
        x = r * jnp.cos(theta)
        y = r * jnp.sin(theta)
        return key, x, y, z
