from jax import numpy as jnp
from jax import jit
from functools import partial

from appletree import randgen
from appletree.plugin import Plugin
from appletree.config import takes_config, Constant, ConstantSet
from appletree.utils import exporter

export, __all__ = exporter(export_self=False)

# These scripts are copied from
# https://github.com/NESTCollaboration/nest/releases/tag/v2.3.7
# and https://github.com/NESTCollaboration/nest/blob/v2.3.7/src/NEST.cpp#L715-L794
# Priors of the distribution is copied from https://arxiv.org/abs/2211.10726
# and https://drive.google.com/file/d/1urVT3htFjIC1pQKyaCcFonvWLt74Kgvn/view
# All variables begins with '_' are expectation values, such as `_Nph`, `_Ne`.


@export
@takes_config(
    ConstantSet(
        name="energy_twohalfnorm",
        default=[
            ["mu", "sigma_pos", "sigma_neg"],
            [[1.0], [0.1], [0.1]],
        ],
        help="Parameterized energy spectrum",
    ),
)
class MonoEnergiesSpectra(Plugin):
    depends_on = ["batch_size"]
    provides = ["energy", "energy_center"]

    @partial(jit, static_argnums=(0, 3))
    def simulate(self, key, parameters, batch_size):
        key, energy = randgen.twohalfnorm(
            key,
            shape=(batch_size, self.energy_twohalfnorm.set_volume),
            **self.energy_twohalfnorm.value
        )
        energy = jnp.clip(energy, a_min=0.0, a_max=jnp.inf)
        energy_center = jnp.broadcast_to(
            self.energy_twohalfnorm.value["mu"], jnp.shape(energy)
        ).astype(float)
        return key, energy, energy_center


@export
@takes_config(
    Constant(
        name="clip_lower_energy",
        type=float,
        default=0.5,
        help="Smallest energy considered in inference",
    ),
    Constant(
        name="clip_upper_energy",
        type=float,
        default=2.5,
        help="Largest energy considered in inference",
    ),
)
class UniformEnergiesSpectra(Plugin):
    depends_on = ["batch_size"]
    provides = ["energy"]

    @partial(jit, static_argnums=(0, 3))
    def simulate(self, key, parameters, batch_size):
        key, energy = randgen.uniform(
            key,
            self.clip_lower_energy.value,
            self.clip_upper_energy.value,
            shape=(batch_size,),
        )
        return key, energy


@export
class TotalQuanta(Plugin):
    depends_on = ["energy"]
    provides = ["_Nq"]
    parameters = ("alpha", "beta")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, energy):
        _Nq = parameters["alpha"] * energy ** parameters["beta"]
        return key, _Nq


@export
@takes_config(
    Constant(
        name="literature_field", type=float, default=23.0, help="Drift field in each literature"
    ),
)
class ThomasImelBox(Plugin):
    depends_on = ["energy"]
    provides = ["ThomasImel"]
    parameters = ("gamma", "delta", "liquid_xe_density")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, energy):
        ThomasImel = jnp.ones(shape=jnp.shape(energy))
        ThomasImel *= parameters["gamma"] * self.literature_field.value ** parameters["delta"]
        ThomasImel *= (parameters["liquid_xe_density"] / 2.9) ** 0.3
        return key, ThomasImel


@export
class QyNR(Plugin):
    depends_on = ["energy", "ThomasImel"]
    provides = ["charge_yield"]
    parameters = ("epsilon", "zeta", "eta")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, energy, ThomasImel):
        charge_yield = 1 / ThomasImel / jnp.sqrt(energy + parameters["epsilon"])
        charge_yield *= 1 - 1 / (1 + (energy / parameters["zeta"]) ** parameters["eta"])
        charge_yield = jnp.clip(charge_yield, 0, jnp.inf)
        return key, charge_yield


@export
class LyNR(Plugin):
    depends_on = ["energy", "_Nq", "charge_yield"]
    provides = ["light_yield"]
    parameters = ("theta", "iota")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, energy, _Nq, charge_yield):
        light_yield = _Nq / energy - charge_yield
        light_yield *= 1 - 1 / (1 + (energy / parameters["theta"]) ** parameters["iota"])
        light_yield = jnp.clip(light_yield, 0, jnp.inf)
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
class MeanExcitonIon(Plugin):
    depends_on = ["ThomasImel", "_Nph", "_Ne"]
    provides = ["_Nex", "_Ni", "nex_ni_ratio", "alf", "elecFrac", "recombProb"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, ThomasImel, _Nph, _Ne):
        _Nex = (-1.0 / ThomasImel) * (
            4.0 * jnp.exp(_Ne * ThomasImel / 4.0) - (_Ne + _Nph) * ThomasImel - 4.0
        )
        _Ni = (4.0 / ThomasImel) * (jnp.exp(_Ne * ThomasImel / 4.0) - 1.0)
        nex_ni_ratio = _Nex / _Ni
        alf = 1.0 / (1.0 + nex_ni_ratio)
        elecFrac = _Ne / (_Nph + _Ne)
        recombProb = 1.0 - (nex_ni_ratio + 1.0) * elecFrac
        return key, _Nex, _Ni, nex_ni_ratio, alf, elecFrac, recombProb


@export
class TrueExcitonIonNR(Plugin):
    depends_on = ["_Nph", "_Ne", "nex_ni_ratio", "alf"]
    provides = ["Ni", "Nex", "Nq"]
    parameters = ("fano_ni", "fano_nex")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, _Nph, _Ne, nex_ni_ratio, alf):
        Nq_mean = _Nph + _Ne
        key, Ni = randgen.truncate_normal(
            key, Nq_mean * alf, jnp.sqrt(parameters["fano_ni"] * Nq_mean * alf), vmin=0
        )
        Ni = Ni.round().astype(int)
        key, Nex = randgen.truncate_normal(
            key,
            Nq_mean * nex_ni_ratio * alf,
            jnp.sqrt(parameters["fano_nex"] * Nq_mean * nex_ni_ratio * alf),
            vmin=0,
        )
        Nex = Nex.round().astype(int)
        Nq = Nex + Ni
        return key, Ni, Nex, Nq


@export
class OmegaNR(Plugin):
    depends_on = ["elecFrac", "recombProb", "Ni"]
    provides = ["omega", "Variance"]
    parameters = ("A", "xi", "omega")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, elecFrac, recombProb, Ni):
        omega = parameters["A"] * jnp.exp(
            -0.5 * (elecFrac - parameters["xi"]) ** 2.0 / (parameters["omega"] ** 2)
        )
        Variance = recombProb * (1.0 - recombProb) * Ni + omega * omega * Ni * Ni
        return key, omega, Variance


@export
class TruePhotonElectronNR(Plugin):
    depends_on = ["recombProb", "Variance", "Ni", "Nq"]
    provides = ["num_photon", "num_electron"]
    parameters = ("alpha2",)

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, recombProb, Variance, Ni, Nq):
        # these parameters will make mean num_electron is just (1. - recombProb) * Ni
        widthCorrection = (
            1.0 - (2.0 / jnp.pi) * parameters["alpha2"] ** 2 / (1.0 + parameters["alpha2"] ** 2)
        ) ** 0.5
        muCorrection = (
            (jnp.sqrt(Variance) / widthCorrection)
            * (parameters["alpha2"] / (1.0 + parameters["alpha2"] ** 2) ** 0.5)
            * 2.0
            * (1.0 / (2.0 * jnp.pi) ** 0.5)
        )

        key, num_electron = randgen.skewnormal(
            key,
            jnp.full(len(recombProb), parameters["alpha2"]),
            (1.0 - recombProb) * Ni - muCorrection,
            jnp.sqrt(Variance) / widthCorrection,
        )
        num_electron = jnp.clip(num_electron.round().astype(int), 0, jnp.inf)
        num_photon = jnp.clip(Nq - num_electron, 0, jnp.inf)
        return key, num_photon, num_electron


@export
@takes_config(
    Constant(
        name="clip_lower_energy",
        type=float,
        default=0.5,
        help="Smallest energy considered in inference",
    ),
    Constant(
        name="clip_upper_energy",
        type=float,
        default=2.5,
        help="Largest energy considered in inference",
    ),
)
class MonoEnergiesClipEff(Plugin):
    """For mono-energy-like yields constrain, we need to filter out the energies out of range.

    The method is set their weights to 0.

    """

    depends_on = ["energy_center"]
    provides = ["eff"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, energy_center):
        mask = energy_center >= self.clip_lower_energy.value
        mask &= energy_center <= self.clip_upper_energy.value
        eff = jnp.where(mask, 1.0, 0.0)
        return key, eff


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
