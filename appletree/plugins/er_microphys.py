from jax import numpy as jnp
from jax import jit
from functools import partial

from appletree import randgen
from appletree.plugin import Plugin
from appletree.utils import exporter

export, __all__ = exporter(export_self=False)


@export
class Quanta(Plugin):
    depends_on = ["energy"]
    provides = ["num_quanta"]
    parameters = (
        "w",
        "fano",
    )

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, energy):
        num_quanta_mean = energy / parameters["w"]
        num_quanta_std = jnp.sqrt(num_quanta_mean * parameters["fano"])
        key, num_quanta = randgen.truncate_normal(key, num_quanta_mean, num_quanta_std, vmin=0)
        return key, num_quanta.round().astype(int)


@export
class IonizationER(Plugin):
    depends_on = ["num_quanta"]
    provides = ["num_ion"]
    parameters = ("nex_ni_ratio",)

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, num_quanta):
        p_ion = 1.0 / (1.0 + parameters["nex_ni_ratio"])
        key, num_ion = randgen.binomial(key, p_ion, num_quanta)
        return key, num_ion


@export
class mTI(Plugin):
    depends_on = ["energy"]
    provides = ["recomb_mean"]
    parameters = ("w", "nex_ni_ratio", "py0", "py1", "py2", "py3", "py4", "field")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, energy):
        ni = energy / parameters["w"] / (1.0 + parameters["nex_ni_ratio"])
        ti = (
            ni
            * parameters["py0"]
            * jnp.exp(-energy / parameters["py1"])
            * parameters["field"] ** parameters["py2"]
            / 4.0
        )
        fd = 1.0 / (1.0 + jnp.exp(-(energy - parameters["py3"]) / parameters["py4"]))
        r = jnp.where(
            ti < 1e-2,
            ti / 2.0 - ti * ti / 3.0,
            1.0 - jnp.log(1.0 + ti) / ti,
        )
        return key, r * fd


@export
class RecombFluct(Plugin):
    depends_on = ["energy"]
    provides = ["recomb_std"]
    parameters = ("rf0", "rf1")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, energy):
        fd_factor = 1.0 - jnp.exp(-energy / parameters["rf1"])
        recomb_std = jnp.clip(parameters["rf0"] * fd_factor, 0, 1.0)
        return key, recomb_std


@export
class TrueRecombER(Plugin):
    depends_on = ["recomb_mean", "recomb_std"]
    provides = ["recomb"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, recomb_mean, recomb_std):
        key, recomb = randgen.truncate_normal(key, recomb_mean, recomb_std, vmin=0.0, vmax=1.0)
        return key, recomb


@export
class RecombinationER(Plugin):
    depends_on = ["num_quanta", "num_ion", "recomb"]
    provides = ["num_photon", "num_electron"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, num_quanta, num_ion, recomb):
        p_not_recomb = 1.0 - recomb
        key, num_electron = randgen.binomial(key, p_not_recomb, num_ion)
        num_photon = num_quanta - num_electron
        return key, num_photon, num_electron
