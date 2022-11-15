from jax import numpy as jnp
from jax import jit
from functools import partial

import appletree
from appletree import interpolation
from appletree import randgen
from appletree.plugin import Plugin
from appletree.config import Map
from appletree.utils import exporter

export, __all__ = exporter(export_self=False)


@export
@appletree.takes_config(
    Map(name='ly_median',
        default='nr_ly_median.json',
        help='Light yield curve from NESTv2'),
)
class LightYield(Plugin):
    depends_on = ['energy']
    provides = ['light_yield']
    parameters = ('t_ly',)

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, energy):
        ly_map = jnp.clip(self.ly_median.map * (1 + parameters['t_ly']), 0, jnp.inf)
        light_yield = interpolation.curve_interpolator(
            energy,
            self.ly_median.coordinate_system,
            ly_map,
        )
        return key, light_yield


@export
class NumberPhoton(Plugin):
    depends_on = ['energy', 'light_yield']
    provides = ['num_photon']

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, energy, light_yield):
        key, num_photon = randgen.poisson(key, light_yield * energy)
        return key, num_photon


@export
@appletree.takes_config(
    Map(name='qy_median',
        default='nr_qy_median.json',
        help='Charge yield curve from NESTv2'),
)
class ChargeYield(Plugin):
    depends_on = ['energy']
    provides = ['charge_yield']
    parameters = ('t_qy',)

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, energy):
        qy_map = jnp.clip(self.qy_median.map * (1 + parameters['t_qy']), 0, jnp.inf)
        charge_yield = interpolation.curve_interpolator(
            energy,
            self.qy_median.coordinate_system,
            qy_map,
        )
        return key, charge_yield


@export
class NumberElectron(Plugin):
    depends_on = ['energy', 'charge_yield']
    provides = ['num_electron']

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, energy, charge_yield):
        key, num_electron = randgen.poisson(key, charge_yield * energy)
        return key, num_electron
