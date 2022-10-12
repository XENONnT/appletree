from functools import partial

from jax import jit
from jax import numpy as jnp

import appletree
from appletree import randgen
from appletree import interpolation
from appletree.plugin import Plugin
from appletree.config import Constant, Map
from appletree.utils import exporter

export, __all__ = exporter()


@export
@appletree.takes_config(
    Constant(name='lower_energy',
        type=float,
        default=0.01,
        help='Energy lower limit simulated in uniformly distribution'),
    Constant(name='upper_energy',
        type=float,
        default=20.,
        help='Energy upper limit simulated in uniformly distribution'),
)
class UniformEnergySpectra(Plugin):
    depends_on = ['batch_size']
    provides = ['energy']

    @partial(jit, static_argnums=(0, 3))
    def simulate(self, key, parameters, batch_size):
        key, energy = randgen.uniform(
            key,
            self.lower_energy.value,
            self.upper_energy.value,
            shape=(batch_size, ),
        )
        return key, energy


@export
@appletree.takes_config(
    Map(name='energy_spectrum',
        default='nr_spectrum.json',
        help='Recoil energy spectrum'),
)
class FixedEnergySpectra(Plugin):
    depends_on = ['batch_size']
    provides = ['energy']

    @partial(jit, static_argnums=(0, 3))
    def simulate(self, key, parameters, batch_size):
        key, p = randgen.uniform(key, 0, 1., shape=(batch_size, ))
        energy = interpolation.curve_interpolator(
            p,
            self.energy_spectrum.coordinate_system,
            self.energy_spectrum.map,
        )
        return key, energy


@export
@appletree.takes_config(
    Constant(name='mono_energy',
        type=float,
        default=2.82,
        help='Mono energy delta function'),
)
class MonoEnergySpectra(Plugin):
    depends_on = ['batch_size']
    provides = ['energy']

    # default energy is Ar37 K shell

    @partial(jit, static_argnums=(0, 3))
    def simulate(self, key, parameters, batch_size):
        energy = jnp.full(batch_size, self.mono_energy.value)
        return key, energy


@export
@appletree.takes_config(
    Constant(name='z_min',
        type=float,
        default=-133.97,
        help='Z lower limit simulated in uniformly distribution'),
    Constant(name='z_max',
        type=float,
        default=-13.35,
        help='Z upper limit simulated in uniformly distribution'),
    Constant(name='r_max',
        type=float,
        default=60.,
        help='Radius upper limit simulated in uniformly distribution'),
)
class PositionSpectra(Plugin):
    depends_on = ['batch_size']
    provides = ['x', 'y', 'z']

    @partial(jit, static_argnums=(0, 3))
    def simulate(self, key, parameters, batch_size):
        key, z = randgen.uniform(key, self.z_min.value, self.z_max.value, shape=(batch_size, ))
        key, r2 = randgen.uniform(key, 0, self.r_max.value**2, shape=(batch_size, ))
        key, theta = randgen.uniform(key, 0, 2*jnp.pi, shape=(batch_size, ))

        r = jnp.sqrt(r2)
        x = r * jnp.cos(theta)
        y = r * jnp.sin(theta)
        return key, x, y, z
