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


@export
@takes_config(
    Constant(
        name="dec_modes",
        type=list,
        default="K1K1,K1L1,K1M1,K1N1,K1O1,L1L1,L1M1,L1N1,M1M1".split(","),
        help="Energy lower limit simulated in uniformly distribution",
    ),
)
class DECEnergy(Plugin):
    depends_on = ["batch_size"]
    provides = ["energy_x", "energy_y"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dec_mode_probabilities = {
            "K1K1":0.7414, "K1L1":0.1880, "K1M1":0.0384, "K1N1":0.0084, "K1O1":0.0013,
            "L1L1":0.0122, "L1M1":0.0049, "L1N1":0.0027, "M1M1":0.0013,}
        dec_mode_energy = {
            "K1K1": 64.62, "K1L1": 37.05, "K1M1": 32.98, "K1N1": 32.11, "K1O1": 31.93,
            "L1L1": 10.04, "L1M1": 6.01, "L1N1": 5.37, "M1M1": 2.05,}
        dec_level_energy = {
            "K1": 33.1694, "L1":5.1881, "L2":4.8521, "M1":1.0721, "M2":0.9305,
            "N1": 0.1864, "N2":0.1301, "O1":0.0136, "O2":0.0038,}

        self.dec_mode_probabilities = jnp.array(
            [dec_mode_probabilities[mode] for mode in self.dec_modes.value]
        )
        self.dec_mode_probabilities /= jnp.sum(self.dec_mode_probabilities)
        self.dec_mode_cdf = jnp.cumsum(self.dec_mode_probabilities)
        self.dec_mode_energy = jnp.array(
            [dec_mode_energy[mode] for mode in self.dec_modes.value]
        )
        self.dec_level_energy = jnp.array(
            [dec_level_energy[mode[:2]] for mode in self.dec_modes.value]
        )

    @partial(jit, static_argnums=(0, 3))
    def simulate(self, key, parameters, batch_size):
        key, uniform = randgen.uniform(key, 0, 1, shape=(batch_size,),)
        index = jnp.searchsorted(self.dec_mode_cdf, uniform)
        energy_x = self.dec_level_energy[index]
        energy_y = self.dec_mode_energy[index] - energy_x

        return key, energy_x, energy_y


@export
class DECQuantaX(er_microphys.Quanta):
    depends_on = ["energy_x"]
    provides = ["num_quanta_x"]

@export
class DECQuantaY(er_microphys.Quanta):
    depends_on = ["energy_y"]
    provides = ["num_quanta_y"]


@export
class DECIonizationX(er_microphys.IonizationER):
    depends_on = ["num_quanta_x"]
    provides = ["num_ion_x"]

@export
class DECIonizationY(er_microphys.IonizationER):
    depends_on = ["num_quanta_y"]
    provides = ["num_ion_y"]


@export
class DECmTIX(er_microphys.mTI):
    depends_on = ["energy_x"]
    provides = ["recomb_mean_x"]

@export
class DECmTIY(er_microphys.mTI):
    depends_on = ["energy_y"]
    provides = ["recomb_mean_y"]


@export
class DECRecombFluctX(er_microphys.RecombFluct):
    depends_on = ["energy_x"]
    provides = ["recomb_std_x"]

@export
class DECRecombFluctY(er_microphys.RecombFluct):
    depends_on = ["energy_y"]
    provides = ["recomb_std_y"]


@export
class DECTrueRecombX(er_microphys.TrueRecombER):
    depends_on = ["recomb_mean_x", "recomb_std_x"]
    provides = ["recomb_x"]

@export
class DECTrueRecombY(er_microphys.TrueRecombER):
    depends_on = ["recomb_mean_y", "recomb_std_y"]
    provides = ["recomb_y"]


@export
class DECRecombComposeConstantR(Plugin):
    depends_on = ["recomb_x", "recomb_y"]
    provides = ["recomb"]
    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, recomb_x, recomb_y):
        recomb = recomb_x * (1.0 - recomb_y) + recomb_y * (1.0 - recomb_x)
        renormalization = recomb + (1.0 - recomb_x) * (1.0 - recomb_y)

        return key, recomb / renormalization

@jit
def running_r_recomb(n1, n2, s1, s2):
    a, b, c, d = n1, n2, s1 / n1, s2 / n2
    
    def recomb(i, state):
        a, b = state
    
        ra = jnp.where(c * a > 0, 1 - jnp.log1p(c * a) / (c * a), 0.0)
        rb = jnp.where(d * b > 0, 1 - jnp.log1p(d * b) / (d * b), 0.0)
        ra_ = ra * (1 - rb)
        rb_ = rb * (1 - ra)
        norm = ra_ + rb_ + (1 - ra) * (1 - rb)

        a_new = a - ra_ / norm
        b_new = b - rb_ / norm
        return a_new, b_new

    init_state = (jnp.float32(a), jnp.float32(b))
    final_a, final_b = lax.fori_loop(0, n1 + n2, recomb, init_state)
    
    return 1 - (final_a + final_b) / (n1 + n2)

v_running_r_recomb = vmap(running_r_recomb)

@export
@takes_config(
    Map(
        name="running_r_scaling",
        default="_running_r_scaling.json",
        help="Scaling factor for the running recombination rate",
    ),
)
class DECRecombComposeRunningR(Plugin):
    depends_on = ["recomb_x", "recomb_y", "num_ion_x", "num_ion_y"]
    provides = ["recomb"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, recomb_x, recomb_y, num_ion_x, num_ion_y):
        scale_x = 10 ** self.running_r_scaling.apply(recomb_x)
        scale_y = 10 ** self.running_r_scaling.apply(recomb_y)
        recomb = v_running_r_recomb(
            num_ion_x, num_ion_y, scale_x, scale_y
        )

        return key, recomb


@export
class DECRecombination(Plugin):
    depends_on = ["num_quanta_x", "num_ion_x", "num_quanta_y", "num_ion_y", "recomb"]
    provides = ["num_photon", "num_electron"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, num_quanta_x, num_ion_x, num_quanta_y, num_ion_y, recomb):
        p_not_recomb = 1.0 - recomb
        key, num_electron = randgen.binomial(key, p_not_recomb, num_ion_x + num_ion_y)
        num_photon = num_quanta_x + num_quanta_y - num_electron
        return key, num_photon, num_electron
