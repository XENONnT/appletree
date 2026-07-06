from jax import numpy as jnp
from jax import scipy as jsp
from jax import jit
from functools import partial

from appletree import randgen
from appletree.plugin import Plugin
from appletree.utils import exporter

export, __all__ = exporter(export_self=False)

# These scripts are copied from
# https://github.com/NESTCollaboration/nest/releases/tag/v2.4.0
# Priors of the distribution is copied from https://arxiv.org/abs/2211.10726
# and https://drive.google.com/file/d/1urVT3htFjIC1pQKyaCcFonvWLt74Kgvn/view
# All variables begins with '_' are expectation values, such as `_Nph`, `_Ne`.


@export
class ExcitonIonRatioER(Plugin):
    depends_on = ["energy"]
    provides = ["nex_ni_ratio", "alf"]
    parameters = ("liquid_xe_density",)

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, energy):
        # The ratio is a constant for liquid xenon
        nex_ni_ratio = (0.067366 + 0.039693 * parameters["liquid_xe_density"]) * jsp.special.erf(
            energy * 0.05
        )
        alf = 1.0 / (1.0 + nex_ni_ratio)
        return key, nex_ni_ratio, alf


@export
class QyER(Plugin):
    depends_on = ["energy", "nex_ni_ratio"]
    provides = ["charge_yield"]
    parameters = (
        "m1",
        "m2",
        "m3",
        "m4",
        "m7",
        "m8",
        "m9",
        "m10",
        "w",
        "field",
        "liquid_xe_density",
    )

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, energy, nex_ni_ratio):
        DENSITY = parameters["liquid_xe_density"]  # 2.8619
        m1 = (
            30.66
            + (parameters["m1"] - 30.66)
            / (1.0 + (parameters["field"] / 73.855) ** 2.0318) ** 0.41883
        )
        m2 = parameters["m2"]
        m3 = jnp.log10(parameters["field"]) * 0.13946236 + parameters["m3"]
        m4 = 1.82217496 + (parameters["m4"] - 1.82217496) / (
            1.0 + (parameters["field"] / 144.65029656) ** -2.80532006
        )
        m5 = 1.0 / parameters["w"] / (1.0 + nex_ni_ratio) - m1
        m7 = 7.02921301 + (parameters["m7"] - 7.02921301) / (
            1.0 + (parameters["field"] / 256.48156448) ** 1.29119251
        )
        m8 = parameters["m8"]
        m9 = parameters["m9"]
        m10 = 0.0508273937 + (parameters["m10"] - 0.0508273937) / (
            1.0 + (parameters["field"] / 139.260460) ** -0.65763592
        )

        charge_yield_beta = jnp.ones(shape=jnp.shape(energy)) * m1
        charge_yield_beta += (m2 - m1) / (1 + (energy / m3) ** m4) ** m9
        charge_yield_beta += jnp.ones(shape=jnp.shape(energy)) * m5
        charge_yield_beta += -m5 / (1 + (energy / m7) ** m8) ** m10
        charge_yield_beta = jnp.clip(charge_yield_beta, 0, jnp.inf)

        coeff_TI = jnp.power(1.0 / DENSITY, 0.3)
        coeff_Ni = jnp.power(1.0 / DENSITY, 1.4)
        coeff_OL = jnp.power(1.0 / DENSITY, -1.7) / jnp.log(
            1.0 + coeff_TI * coeff_Ni * jnp.power(DENSITY, 1.7)
        )
        charge_yield_beta = charge_yield_beta * (
            coeff_OL
            * jnp.log(1.0 + coeff_TI * coeff_Ni * jnp.power(DENSITY, 1.7))
            * jnp.power(DENSITY, -1.7)
        )

        return key, charge_yield_beta


@export
class LyER(Plugin):
    depends_on = ["charge_yield"]
    provides = ["light_yield"]
    parameters = ("w",)

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, charge_yield):
        light_yield = 1.0 / parameters["w"] - charge_yield
        light_yield = jnp.maximum(light_yield, 0.0)
        return key, light_yield


@export
class MeanNphNe(Plugin):
    depends_on = ["light_yield", "charge_yield", "energy"]
    provides = ["_Nph", "_Ne"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, light_yield, charge_yield, energy):
        _Nph = light_yield * energy
        _Ne = charge_yield * energy
        return key, _Nph, _Ne


@export
class MeanExcitonIonER(Plugin):
    depends_on = ["nex_ni_ratio", "_Nph", "_Ne"]
    provides = ["elecFrac", "recombProb"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, nex_ni_ratio, _Nph, _Ne):
        elecFrac = _Ne / (_Nph + _Ne)
        recombProb = 1.0 - (nex_ni_ratio + 1.0) * elecFrac
        recombProb = jnp.maximum(recombProb, 0.0)
        return key, elecFrac, recombProb


@export
class FanoFactor(Plugin):
    depends_on = ["_Nph", "_Ne"]
    provides = [
        "fano_nq",
    ]
    parameters = ("field", "delta_f", "liquid_xe_density")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, _Nph, _Ne):
        # Mimicking the behavior of NEST v2.4.0
        # negative 0.0015 restores https://arxiv.org/abs/2211.10726v3 Eq. 8
        sign = jnp.sign(parameters["delta_f"])
        abs_delta_f = jnp.abs(parameters["delta_f"])

        # Fano factors in LXe for ER if delta_f is positive
        fano_nq = (sign + 1.0) / 2.0 * jnp.ones(len(_Ne)) * abs_delta_f

        # Fano factors in LXe for ER if delta_f is negative
        fano_nq_const = (
            0.12707
            - 0.029623 * parameters["liquid_xe_density"]
            - 0.0057042 * parameters["liquid_xe_density"] ** 2
            + 0.0015957 * parameters["liquid_xe_density"] ** 3
        )
        fano_nq += (
            (1.0 - sign)
            / 2.0
            * (fano_nq_const + abs_delta_f * jnp.sqrt((_Nph + _Ne) * parameters["field"]))
        )

        return key, fano_nq


@export
class TrueExcitonIonER(Plugin):
    depends_on = ["_Nph", "_Ne", "fano_nq", "alf"]
    provides = ["Ni", "Nex", "Nq"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, _Nph, _Ne, fano_nq, alf):
        Nq_mean = _Nph + _Ne
        key, Nq = randgen.truncate_normal(key, Nq_mean, jnp.sqrt(fano_nq * Nq_mean), 
                                          vmin=0.0, vmax=jnp.inf)
        Nq = Nq.round().astype(int)
        key, Ni = randgen.binomial(key, alf, Nq)
        Nex = Nq - Ni
        Nex = jnp.clip(Nex.round().astype(int), 0, Nq)
        Ni = jnp.clip(Ni.round().astype(int), 0, Nq)
        return key, Ni, Nex, Nq


@export
class OmegaER(Plugin):
    depends_on = ["elecFrac", "recombProb", "Ni", "_Ne", "_Nph"]
    provides = ["omega", "Variance"]
    parameters = ("A_er", "xi_er", "omega_er", "alpha3_er", "field")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, elecFrac, recombProb, Ni, _Ne, _Nph):
        # was A in previous version, be advised
        ampl = (
            0.086036
            + (parameters["A_er"] - 0.086036)
            / (1.0 + (parameters["field"] / 295.2) ** 251.6) ** 0.0069114
        )
        sqrt2 = jnp.sqrt(2.0)

        # Skewd normal normalization
        skew = parameters["alpha3_er"]
        wide = parameters["omega_er"]
        cntr = parameters["xi_er"]
        mode = cntr + jnp.sqrt(2.0 / jnp.pi) * skew * wide / jnp.sqrt(1.0 + skew * skew)

        norm = 1.0 / (
            jnp.exp(-0.5 * (mode - cntr) ** 2 / (wide * wide))
            * (1.0 + jsp.special.erf(skew * (mode - cntr) / (wide * sqrt2)))
        )  # makes sure omega never exceeds ampl
        omega = (norm
            * ampl
            * jnp.exp(-0.5 * (elecFrac - cntr) ** 2 / (wide * wide))
            * (1.0 + jsp.special.erf(skew * (elecFrac - cntr) / (wide * sqrt2)))
        )
        omega = jnp.maximum(omega, 0.0)
        Variance = recombProb * (1.0 - recombProb) * Ni + omega * omega * Ni * Ni
        return key, omega, Variance


@export
class TruePhotonElectronER(Plugin):
    depends_on = ["recombProb", "Variance", "Ni", "Nq"]
    provides = ["num_photon", "num_electron"]
    parameters = ("field",)

    @partial(jit, static_argnums=(0,))
    def simulate(
        self,
        key,
        parameters,
        recombProb,
        Variance,
        Ni,
        Nq,
    ):
        key, num_electron = randgen.normal(key, (1 - recombProb) * Ni, jnp.sqrt(Variance))
        num_electron = jnp.clip(num_electron.round().astype(int), 0, Ni)  # 16.02.2026 jnp.inf ->Ni
        # num_electron = jnp.clip(num_electron.round().astype(int), 0, Nq)
        num_photon = jnp.clip(Nq - num_electron, 0, jnp.inf)
        return key, num_photon, num_electron


@export
class BandEnergiesClipEff(Plugin):
    """For band-like yields constrain, we only need a placeholder here.

    Because BandEnergySpectra has already selected energy for us.

    """

    depends_on = ["energy"]
    provides = ["eff"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, energy):
        eff = jnp.ones(len(energy))
        return key, eff
