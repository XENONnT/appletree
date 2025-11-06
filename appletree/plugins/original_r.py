from re import S
from typing import final
from jax import numpy as jnp
from jax import jit, vmap, lax
from functools import partial

from appletree import randgen
from appletree.plugin import Plugin
from appletree.plugins import er_microphys
from appletree.config import takes_config, Constant, Map
from appletree.utils import exporter
#from simulation.composite.composite_utils import running_uncertainty_plot

export, __all__ = exporter(export_self=False)

"""
# For Ar37
@takes_config(
    Constant(name="Auger", type=float, default=2.33, help="Mono energy of KLL Auger electron"),
    Constant(name="L1", type=float, default=0.2, help="Mono energy of L-shell vacancy"),
    Constant(name="L2", type=float, default=0.27, help="Mono energy of L-shell vacancy"),
)
"""
@export
@takes_config(
    Constant(name="ICe", type=float, default=129.3, help="Mono energy of internal conversion electron"),
    Constant(name="Ka1", type=float, default=29.8, help="Mono energy of K-shell vacancy"),
    Constant(name="L", type=float, default=4.8, help="Mono energy of L-shell vacancy"),
)
class ECEnergy(Plugin):
    depends_on = ["batch_size"]
    provides = ["energy_x", "energy_y", "energy_z"]

    @partial(jit, static_argnums=(0, 3))
    def simulate(self, key, parameters, batch_size):
        energy_x = jnp.full(batch_size, self.ICe.value)
        energy_y = jnp.full(batch_size, self.Ka1.value)
        energy_z = jnp.full(batch_size, self.L.value)
        return key, energy_x, energy_y, energy_z


@export
class DefaultQuantaX(er_microphys.Quanta):
    depends_on = ["energy_x"]
    provides = ["num_quanta_x"]
@export
class DefaultQuantaY(er_microphys.Quanta):
    depends_on = ["energy_y"]
    provides = ["num_quanta_y"]
@export
class DefaultQuantaZ(er_microphys.Quanta):
    depends_on = ["energy_z"]
    provides = ["num_quanta_z"]


@export
class DefaultIonizationX(er_microphys.IonizationER):
    depends_on = ["num_quanta_x"]
    provides = ["num_ion_x"]
@export
class DefaultIonizationY(er_microphys.IonizationER):
    depends_on = ["num_quanta_y"]
    provides = ["num_ion_y"]
@export
class DefaultIonizationZ(er_microphys.IonizationER):
    depends_on = ["num_quanta_z"]
    provides = ["num_ion_z"]


@export
class DefaultmTIX(er_microphys.mTI):
    depends_on = ["energy_x"]
    provides = ["recomb_mean_x"]
@export
class DefaultmTIY(er_microphys.mTI):
    depends_on = ["energy_y"]
    provides = ["recomb_mean_y"]
@export
class DefaultmTIZ(er_microphys.mTI):
    depends_on = ["energy_z"]
    provides = ["recomb_mean_z"]


@export
class DefaultRecombFluctX(er_microphys.RecombFluct):
    depends_on = ["energy_x"]
    provides = ["recomb_std_x"]
@export
class DefaultRecombFluctY(er_microphys.RecombFluct):
    depends_on = ["energy_y"]
    provides = ["recomb_std_y"]
@export
class DefaultRecombFluctZ(er_microphys.RecombFluct):
    depends_on = ["energy_z"]
    provides = ["recomb_std_z"]


@export
class DefaultTrueRecombX(er_microphys.TrueRecombER):
    depends_on = ["recomb_mean_x", "recomb_std_x"]
    provides = ["recomb_x"]
@export
class DefaultTrueRecombY(er_microphys.TrueRecombER):
    depends_on = ["recomb_mean_y", "recomb_std_y"]
    provides = ["recomb_y"]
@export
class DefaultTrueRecombZ(er_microphys.TrueRecombER):
    depends_on = ["recomb_mean_z", "recomb_std_z"]
    provides = ["recomb_z"]


@export
class RecombinationOriginalR(Plugin):
    depends_on = ["num_quanta_x", "num_ion_x", "num_quanta_y", "num_ion_y", "num_quanta_z", "num_ion_z", "recomb_x", "recomb_y", "recomb_z"]
    provides = ["num_photon", "num_electron"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, num_quanta_x, num_ion_x, num_quanta_y, num_ion_y, num_quanta_z, num_ion_z, recomb_x, recomb_y, recomb_z):
        p_not_recomb_x = 1.0 - recomb_x
        p_not_recomb_y = 1.0 - recomb_y
        p_not_recomb_z = 1.0 - recomb_z
        key, num_electron_x = randgen.binomial(key, p_not_recomb_x, num_ion_x)
        key, num_electron_y = randgen.binomial(key, p_not_recomb_y, num_ion_y)
        key, num_electron_z = randgen.binomial(key, p_not_recomb_z, num_ion_z)
        num_electron = num_electron_x + num_electron_y + num_electron_z
        num_photon = num_quanta_x + num_quanta_y + num_quanta_z - num_electron
        return key, num_photon, num_electron
