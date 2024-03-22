from jax import numpy as jnp
from jax import scipy as jsp
from jax import jit
from functools import partial
import scipy

from appletree import randgen
from appletree.plugin import Plugin
from appletree.utils import exporter


export, __all__ = exporter(export_self=False)


def atanh_clip(x):
    return jnp.arctanh((x + 1) % 2 - 1)


def n_photon_mean_model0(num_quanta, p_mu_0):
    return (
        jnp.sin(jsp.special.erf(atanh_clip(jnp.sqrt(jnp.arcsinh(num_quanta)))))
        * num_quanta
        / p_mu_0
    )


def n_photon_std_model0(num_quanta, p_sigma_0):
    return (num_quanta * p_sigma_0) + jnp.log2((jnp.log2(jnp.log2(jnp.arcsinh(num_quanta)))) ** 6)


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
class Emission(Plugin):
    """A symbolic-regression-based empirical simulation for microphysics aiming at parameter number
    reduction.

    Wrapped up IonizationER + mTI + RecombFluct + TrueRecombER + RecombinationER in traditional ER
    microphysics.

    """

    depends_on = ["num_quanta"]
    provides = ["num_photon", "num_electron"]
    parameters = ("p_mu_0", "p_sigma_0")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, num_quanta):
        n_photon_mean = n_photon_mean_model0(num_quanta, parameters["p_mu_0"])
        n_photon_std = n_photon_std_model0(num_quanta, parameters["p_sigma_0"])
        key, _num_photon = randgen.truncate_normal(
            key, n_photon_mean, n_photon_std, vmin=0, vmax=num_quanta
        )
        num_photon = _num_photon.round().astype(int)
        num_electron = num_quanta - num_photon
        return key, num_photon, num_electron
