from jax import numpy as jnp
from jax import jit, vmap, lax
from functools import partial

from appletree import randgen
from appletree.plugin import Plugin
from appletree.plugins import er_nestv2
from appletree.config import takes_config, Constant, Map
from appletree.utils import exporter

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
class DECExcitonIonRatioX(er_nestv2.ExcitonIonRatioER):
    depends_on = ["energy_x"]
    provides = ["nex_ni_ratio_x", "alf_x"]

@export
class DECExcitonIonRatioY(er_nestv2.ExcitonIonRatioER):
    depends_on = ["energy_y"]
    provides = ["nex_ni_ratio_y", "alf_y"]

@export
class DECQyX(er_nestv2.QyER):
    depends_on = ["energy_x", "nex_ni_ratio_x"]
    provides = ["charge_yield_x"]

@export
class DECQyY(er_nestv2.QyER):
    depends_on = ["energy_y", "nex_ni_ratio_y"]
    provides = ["charge_yield_y"]

@export
class DECLyX(er_nestv2.LyER):
    depends_on = ["charge_yield_x"]
    provides = ["light_yield_x"]

@export
class DECLyY(er_nestv2.LyER):
    depends_on = ["charge_yield_y"]
    provides = ["light_yield_y"]

@export
class DECMeanNphNeX(er_nestv2.MeanNphNe):
    depends_on = ["light_yield_x", "charge_yield_x", "energy_x"]
    provides = ["_Nph_x", "_Ne_x"]

@export
class DECMeanNphNeY(er_nestv2.MeanNphNe):
    depends_on = ["light_yield_y", "charge_yield_y", "energy_y"]
    provides = ["_Nph_y", "_Ne_y"]

@export
class DECMeanExcitonIonX(er_nestv2.MeanExcitonIonER):
    depends_on = ["nex_ni_ratio_x", "_Nph_x", "_Ne_x"]
    provides = ["elecFrac_x", "recombProb_x"]

@export
class DECMeanExcitonIonY(er_nestv2.MeanExcitonIonER):
    depends_on = ["nex_ni_ratio_y", "_Nph_y", "_Ne_y"]
    provides = ["elecFrac_y", "recombProb_y"]

@export
class DECFanoFactorX(er_nestv2.FanoFactor):
    depends_on = ["_Nph_x", "_Ne_x"]
    provides = ["fano_nq_x"]

@export
class DECFanoFactorY(er_nestv2.FanoFactor):
    depends_on = ["_Nph_y", "_Ne_y"]
    provides = ["fano_nq_y"]

@export
class DECTrueExcitonIonX(er_nestv2.TrueExcitonIonER):
    depends_on = ["_Nph_x", "_Ne_x", "fano_nq_x", "alf_x"]
    provides = ["Ni_x", "Nex_x", "Nq_x"]

@export
class DECTrueExcitonIonY(er_nestv2.TrueExcitonIonER):
    depends_on = ["_Nph_y", "_Ne_y", "fano_nq_y", "alf_y"]
    provides = ["Ni_y", "Nex_y", "Nq_y"]

@export
class DECTotalProperties(Plugin):
    depends_on = ["Ni_x", "Ni_y", "Nq_x", "Nq_y",
                  "_Ne_x", "_Ne_y", "_Nph_x", "_Nph_y",
                  "energy_x", "energy_y",
                  ]
    provides = ["elecFrac", "Ni", "Nq", "_Ne", "_Nph", "energy"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, Ni_x, Ni_y, Nq_x, Nq_y, _Ne_x, _Ne_y, _Nph_x, _Nph_y, energy_x, energy_y):
        elecFrac = (_Ne_x + _Ne_y) / (_Nph_x + _Nph_y + _Ne_x + _Ne_y)
        return key, elecFrac, Ni_x + Ni_y, Nq_x + Nq_y, _Ne_x + _Ne_y, _Nph_x + _Nph_y, energy_x + energy_y

@export
class DECRecombComposeConstantR(Plugin):
    depends_on = ["recombProb_x", "recombProb_y"]
    provides = ["recombProb"]
    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, recombProb_x, recombProb_y):
        recomb = recombProb_x * (1.0 - recombProb_y) + recombProb_y * (1.0 - recombProb_x)
        renormalization = recomb + (1.0 - recombProb_x) * (1.0 - recombProb_y)

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
    depends_on = ["recombProb_x", "recombProb_y", "Ni_x", "Ni_y"]
    provides = ["recombProb"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, recombProb_x, recombProb_y, Ni_x, Ni_y):
        scale_x = 10 ** self.running_r_scaling.apply(recombProb_x)
        scale_y = 10 ** self.running_r_scaling.apply(recombProb_y)
        recomb = v_running_r_recomb(
            Ni_x, Ni_y, scale_x, scale_y
        )

        return key, recomb

@export
class DECOmegaER(er_nestv2.OmegaER):
    depends_on = ["elecFrac", "recombProb", "Ni", "_Ne", "_Nph"]
    provides = ["omega", "Variance"]

@export
class DECTruePhotonElectronER(er_nestv2.TruePhotonElectronER):
    depends_on = ["recombProb", "Variance", "Ni", "Nq", "energy"]
    provides = ["num_photon", "num_electron"]
