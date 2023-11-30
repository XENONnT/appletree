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
        name="s1_correction",
        default="_s1_correction.json",
        help="S1 light collection efficiency correction",
    ),
)
class S1CorrectionTrue(Plugin):
    depends_on = ["x", "y", "z"]
    provides = ["s1_correction_true"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, x, y, z):
        pos_true = jnp.stack([x, y, z]).T
        s1_correction_true = self.s1_correction.apply(pos_true)
        return key, s1_correction_true

@export
@takes_config(
    Map(
        name="s1_correction",
        default="_s1_correction.json",
        help="S1 light collection efficiency correction",
    ),
)
class S1CorrectionRec(Plugin):
    depends_on = ["rec_x", "rec_y", "rec_z"]
    provides = ["s1_correction_rec"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, rec_x, rec_y, rec_z):
        pos_rec = jnp.stack([rec_x, rec_y, rec_z]).T
        s1_correction_rec = self.s1_correction.apply(pos_rec)
        return key, s1_correction_rec


@export
@takes_config(
    Map(
        name="s2_correction",
        default="_s2_correction.json",
        help="S2 light collection efficiency correction",
    ),
)
class S2CorrectionTrue(Plugin):
    depends_on = ["x", "y"]
    provides = ["s2_correction_true"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, x, y):
        pos_true = jnp.stack([x, y]).T
        s2_correction_true = self.s2_correction.apply(pos_true)
        return key, s2_correction_true

@export
@takes_config(
    Map(
        name="s2_correction",
        default="_s2_correction.json",
        help="S2 light collection efficiency correction",
    ),
)
class S2CorrectionRec(Plugin):
    depends_on = ["rec_x", "rec_y"]
    provides = ["s2_correction_rec"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, rec_x, rec_y):
        pos_rec = jnp.stack([rec_x, rec_y]).T
        s2_correction_rec = self.s2_correction.apply(pos_rec)
        return key, s2_correction_rec

@export
class PhotonDetection(Plugin):
    depends_on = ["num_photon", "s1_correction_true"]
    provides = ["num_s1_phd"]
    parameters = ("g1", "p_dpe")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, num_photon, s1_correction_true):
        g1_true_no_dpe = jnp.clip(
            parameters["g1"] * s1_correction_true / (1.0 + parameters["p_dpe"]), 0, 1.0
        )
        key, num_s1_phd = randgen.binomial(key, g1_true_no_dpe, num_photon)
        return key, num_s1_phd


@export
class S1PE(Plugin):
    depends_on = ["num_s1_phd"]
    provides = ["num_s1_pe"]
    parameters = ("p_dpe",)

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, num_s1_phd):
        key, num_s1_dpe = randgen.binomial(key, parameters["p_dpe"], num_s1_phd)
        num_s1_pe = num_s1_dpe + num_s1_phd
        return key, num_s1_pe


@export
@takes_config(
    Map(name="elife", default="_elife.json", help="Electron lifetime correction"),
)
class DriftLoss(Plugin):
    depends_on = ["z"]
    provides = ["drift_survive_prob"]
    parameters = ("drift_velocity", "elife_sigma")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, z):
        key, p = randgen.uniform(key, 0, 1.0, shape=jnp.shape(z))
        lifetime = self.elife.apply(p) * (1 + parameters["elife_sigma"])
        drift_survive_prob = jnp.exp(-jnp.abs(z) / parameters["drift_velocity"] / lifetime)
        return key, drift_survive_prob


@export
class ElectronDrifted(Plugin):
    depends_on = ["num_electron", "drift_survive_prob"]
    provides = ["num_electron_drifted"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, num_electron, drift_survive_prob):
        key, num_electron_drifted = randgen.binomial(key, drift_survive_prob, num_electron)
        return key, num_electron_drifted


@takes_config(
    Map(name="gas_gain", default="_gas_gain.json", help="Gas gain (x,y) dependence"),
)
@export
class S2PE(Plugin):
    depends_on = ["num_electron_drifted", "s2_correction_true", "x", "y"]
    provides = ["num_s2_pe"]
    parameters = ("g2", "gas_gain")

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, num_electron_drifted, s2_correction_true, x, y):
        pos_true = jnp.stack([x, y]).T
        gas_gain = self.gas_gain.apply((pos_true))
        extraction_eff = parameters["g2"] * s2_correction_true / gas_gain

        key, num_electron_extracted = randgen.binomial(key, extraction_eff, num_electron_drifted)

        mean_s2_pe = num_electron_extracted * gas_gain
        key, num_s2_pe = randgen.truncate_normal(key, mean_s2_pe, jnp.sqrt(mean_s2_pe), vmin=0)

        return key, num_s2_pe
