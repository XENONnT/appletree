from jax import numpy as jnp
from jax import jit
from functools import partial

from appletree import randgen
from appletree.plugin import Plugin
from appletree.config import takes_config, Map
from appletree.utils import exporter

export, __all__ = exporter(export_self=False)

@export
@takes_config(
    Map(
        name="ly_median",
        method="LERP",
        default="_er_ly.json",
        help="Light yield curve",
    ),
)

class LightYield(Plugin):
    depends_on = ["energy"]
    provides = ["light_yield"]
    parameters = ("t_er_ly",)

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, energy):
        light_yield = self.ly_median.apply(energy) * (1.0 + parameters["t_er_ly"])
        light_yield = jnp.clip(light_yield, 0, jnp.inf)
        return key, light_yield


@export
class NumberPhoton(Plugin):
    depends_on = ["energy", "light_yield"]
    provides = ["num_photon"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, energy, light_yield):
        key, num_photon = randgen.poisson(key, light_yield * energy)
        return key, num_photon


@export
@takes_config(
    Map(
        name="qy_median",
        method="LERP",
        default="_er_qy.json",
        help="Charge yield curve",
    ),
)
class ChargeYield(Plugin):
    depends_on = ["energy"]
    provides = ["charge_yield"]
    parameters = ("t_er_qy",)

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, energy):
        charge_yield = self.qy_median.apply(energy) * (1.0 + parameters["t_er_qy"])
        charge_yield = jnp.clip(charge_yield, 0, jnp.inf)
        return key, charge_yield


@export
class NumberElectron(Plugin):
    depends_on = ["energy", "charge_yield"]
    provides = ["num_electron"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, energy, charge_yield):
        key, num_electron = randgen.poisson(key, charge_yield * energy)
        return key, num_electron
