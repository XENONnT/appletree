from jax import numpy as jnp
from jax import jit
from functools import partial

import appletree
from appletree import randgen
from appletree.plugin import Plugin
from appletree.config import Constant, ConstantSet
from appletree.utils import exporter

export, __all__ = exporter(export_self=False)

# These scripts are copied from
# https://github.com/NESTCollaboration/nest/releases/tag/v2.3.7
# and https://github.com/NESTCollaboration/nest/blob/v2.3.7/src/NEST.cpp#L715-L794
# Priors of the distribution is copied from https://arxiv.org/abs/2211.10726
# and https://drive.google.com/file/d/1urVT3htFjIC1pQKyaCcFonvWLt74Kgvn/view


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
class MonoEnergiesSpectra(Plugin):
    depends_on = ['batch_size']
    provides = ['energy', 'energy_center']

    @partial(jit, static_argnums=(0, 3))
    def simulate(self, key, parameters, batch_size):
        key, energy = randgen.twohalfnorm(
            key,
            shape=(batch_size, self.energy_twohalfnorm.set_volume),
            **self.energy_twohalfnorm.value)
        energy = jnp.clip(energy, a_min=0., a_max=jnp.inf)
        energy_center = jnp.broadcast_to(
            self.energy_twohalfnorm.value['mu'], jnp.shape(energy)).astype(float)
        return key, energy, energy_center


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
class UniformEnergiesSpectra(Plugin):
    depends_on = ['batch_size']
    provides = ['energy']

    @partial(jit, static_argnums=(0, 3))
    def simulate(self, key, parameters, batch_size):
        key, energy = randgen.uniform(
            key,
            self.clip_lower_energy.value,
            self.clip_upper_energy.value,
            shape=(batch_size, ),
        )
        return key, energy


@export
class TotalQuanta(Plugin):
    depends_on = ['energy']
    provides = ['Nq']
    parameters = ('alpha', 'beta')

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, energy):
        Nq = parameters['alpha'] * energy ** parameters['beta']
        return key, Nq


@export
@appletree.takes_config(
    Constant(name='literature_field',
        type=float,
        default=23.0,
        help='Drift field in each literature'),
)
class TIB(Plugin):
    depends_on = ['energy']
    provides = ['ThomasImel']
    parameters = ('gamma', 'delta', 'liquid_xe_density')

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, energy):
        ThomasImel = jnp.ones(shape=jnp.shape(energy))
        ThomasImel *= parameters['gamma'] * self.literature_field.value ** parameters['delta']
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
        charge_yield = jnp.clip(charge_yield, 0, jnp.inf)
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
        light_yield = jnp.clip(light_yield, 0, jnp.inf)
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
class MonoEnergiesClipEff(Plugin):
    """
    For mono-energy-like yields constrain,
    we need to filter out the energies out of range.
    The method is set their weights to 0.
    """

    depends_on = ['energy_center']
    provides = ['eff']

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, energy_center):
        mask = energy_center >= self.clip_lower_energy.value
        mask &= energy_center <= self.clip_upper_energy.value
        eff = jnp.where(mask, 1., 0.)
        return key, eff


@export
class BandEnergiesClipEff(Plugin):
    """
    For band-like yields constrain,
    we only need a placeholder here.
    Because BandEnergySpectra has already selected energy for us.
    """

    depends_on = ['energy']
    provides = ['eff']

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, energy):
        eff = jnp.ones(len(energy))
        return key, eff
