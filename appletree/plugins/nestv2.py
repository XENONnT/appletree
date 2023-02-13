from jax import numpy as jnp
from jax import jit
from functools import partial

import appletree
from appletree import randgen
from appletree.plugin import Plugin
from appletree.config import Constant, ConstantSet
from appletree.utils import exporter

export, __all__ = exporter(export_self=False)


@export
@appletree.takes_config(
    ConstantSet(
        name='energy_twohalfnorm',
        default=[
            ['mu', 'sigma_pos', 'sigma_neg'],
            [[1.0], [0.1], [0.1]],
        ],
        help='Parameterized energy spectrum'),
)
class ParameterizedEnergySpectra(Plugin):
    depends_on = ['batch_size']
    provides = ['energy']

    @partial(jit, static_argnums=(0, 3))
    def simulate(self, key, parameters, batch_size):
        key, energy = randgen.twohalfnorm(
            key,
            shape=(batch_size, self.energy_twohalfnorm.set_volume),
            **self.energy_twohalfnorm.value)
        return key, energy


@export
class Nq(Plugin):
    depends_on = ['energy']
    provides = ['Nq']
    parameters = ('alpha', 'beta')

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, energy):
        Nq = parameters['alpha'] * energy ** parameters['beta']
        return key, Nq


@export
class TIB(Plugin):
    depends_on = ['energy']
    provides = ['ThomasImel']
    parameters = ('gamma', 'delta', 'field', 'liquid_xe_density')

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, energy):
        ThomasImel = jnp.ones(shape=jnp.shape(energy))
        ThomasImel *= parameters['gamma'] * parameters['field'] ** parameters['delta']
        ThomasImel *= (parameters['liquid_xe_density'] / 2.9) ** 0.3
        return key, ThomasImel


@export
class Qy(Plugin):
    depends_on = ['energy', 'ThomasImel']
    provides = ['charge_yield']
    parameters = ('epsilon', 'zeta', 'eta')

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, energy, ThomasImel):
        charge_yield = 1 / ThomasImel / jnp.sqrt(energy + parameters['epsilon'])
        charge_yield *= (1 - 1 / (1 + (energy / parameters['zeta']) ** parameters['eta']))
        return key, charge_yield


@export
class Ly(Plugin):
    depends_on = ['energy', 'Nq', 'charge_yield']
    provides = ['light_yield']
    parameters = ('theta', 'iota')

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, energy, Nq, charge_yield):
        light_yield = Nq / energy - charge_yield
        light_yield *= (1 - 1 / (1 + (energy / parameters['theta']) ** parameters['iota']))
        return key, light_yield


@export
@appletree.takes_config(
    Constant(name='clip_lower_energy',
        type=float,
        default=0.5,
        help='Smallest energy considered in inference'),
    Constant(name='clip_upper_energy',
        type=float,
        default=2.5,
        help='Largest energy considered in inference'),
)
class ClipEff(Plugin):
    depends_on = ['energy']
    provides = ['eff']

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, energy):
        eff = jnp.where(
            (energy >= self.clip_lower_energy.value) & (energy <= self.clip_upper_energy.value),
            1.0, 0.0)
        return key, eff
