from jax import numpy as jnp
from jax import scipy as jsp
from jax import jit
from functools import partial

from appletree import randgen
from appletree.plugin import Plugin
from appletree.utils import exporter


export, __all__ = exporter(export_self=False)


def n_photon_mean_model(num_quanta, p_mu_0=0.9982161, p_mu_1=0.418):
    """Mean of Gaussian model bridging quanta number to photon number. \begin{aligned} \mu= &
    p_{\mu_0}\left( \sqrt{x_0}+ \operatorname{asinh}^3\left(\sin ^3.

    \left(\log \left(x_0+p_{\mu_0}\right)\right)\right)-1/0.249\right)^2 \\ &
    -\operatorname{atan}^2\left(\operatorname{erfc}\left(\sin \left(\log
    \left(x_0+p_{\mu_0}\right)\right)\right)\right) & x_0 \log \left(p_{\mu_1} \tanh \left(\sin
    \left(\sin \left(\tan \left( \sinh \left(\sin \left(\log
    \left(x_0+1\right)\right)\right)\right)\right) \right)\right)+1\right)^3 \end{aligned}

    """
    result = (
        p_mu_0
        * (
            jnp.sqrt(num_quanta)
            + jnp.arcsinh(jnp.sin(jnp.log(num_quanta + p_mu_0)) ** 3)
            - 1 / 0.249
        )
        ** 2
        - jnp.log(jsp.special.erfc(jnp.sin(jnp.log(num_quanta + p_mu_0))) + 1)
        - (
            num_quanta
            * jnp.log(
                p_mu_1
                * jnp.tanh(jnp.sin(jnp.sin(jnp.tan(jnp.sinh(jnp.sin(jnp.log(num_quanta + 1)))))))
                + 1
            )
            ** 3
        )
    )
    return result


def n_photon_std_model(num_quanta, p_sigma_0=0.4):
    """Standard deviation of Gaussian model bridging quanta number to photon number.

    \sigma=x_0^{p_{\sigma_0}}-\sinh \left(\cosh \left(\sin \left(
    \operatorname{asinh}\left(x_0\right)\right)\right)\right)

    """
    result = num_quanta**p_sigma_0 - jnp.sinh(jnp.cosh(jnp.sin(jnp.arcsinh(num_quanta))))
    return result


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
    parameters = ("p_mu_0", "p_mu_1", "p_sigma_0")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, num_quanta):
        n_photon_mean = n_photon_mean_model(
            num_quanta,
            parameters["p_mu_0"],
            parameters["p_mu_1"],
        )
        n_photon_std = n_photon_std_model(num_quanta, parameters["p_sigma_0"])
        key, _num_photon = randgen.truncate_normal(
            key, n_photon_mean, n_photon_std, vmin=0, vmax=num_quanta
        )
        num_photon = _num_photon.round().astype(int)
        num_electron = num_quanta - num_photon
        return key, num_photon, num_electron
