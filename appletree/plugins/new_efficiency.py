from jax import numpy as jnp
from jax import jit
from functools import partial

from appletree.plugin import Plugin
from appletree.config import takes_config, SigmaMap
from appletree.utils import exporter

from applefiles import aptext

export, __all__ = exporter(export_self=False)


@export
class S2Threshold(Plugin):
    depends_on = ["s2_area"]
    provides = ["acc_s2_threshold"]
    parameters = ("s2_threshold",)

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, s2_area):
        return key, jnp.where(s2_area > parameters["s2_threshold"], 1.0, 0)


class S1ReconEffNHits(aptext.acceptance.efficiency_3f.S1ReconEffNHits):
    depends_on = ["s1_n_hits"]
    provides = ["acc_s1_recon_eff"]


@export
@takes_config(
    SigmaMap(
        name="s1_cut_acc",
        method="LERP",
        default=["_s1_cut_acc.json", "_s1_cut_acc.json", "_s1_cut_acc.json"],
        help="S1 cut acceptance",
    ),
)
class S1CutAccept(Plugin):
    depends_on = ["s1_area"]
    provides = ["cut_acc_s1"]
    # parameters = ("s1_cut_acc_sigma",)

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, s1_area):
        cut_acc_s1 = self.s1_cut_acc.apply(s1_area, parameters)
        cut_acc_s1 = jnp.clip(cut_acc_s1, 0.0, 1.0)
        return key, cut_acc_s1


@export
@takes_config(
    SigmaMap(
        name="s2_cut_acc",
        method="LERP",
        default=["_s2_cut_acc.json", "_s2_cut_acc.json", "_s2_cut_acc.json", "s2_cut_acc_sigma"],
        help="S2 cut acceptance",
    ),
)
class S2CutAccept(Plugin):
    depends_on = ["s2_area"]
    provides = ["cut_acc_s2"]
    # parameters = ("s2_cut_acc_sigma",)

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, s2_area):
        cut_acc_s2 = self.s2_cut_acc.apply(s2_area, parameters)
        cut_acc_s2 = jnp.clip(cut_acc_s2, 0.0, 1.0)
        return key, cut_acc_s2


class FiducialVolumeCylinderAccept(aptext.acceptance.cut_fv.FiducialVolumeCylinderAccept):
    depends_on = ["rec_z", "rec_r"]
    provides = ["acc_fv_cut"]


class AntiCorrelationEfficiency(aptext.acceptance.efficiency_2f.AntiCorrelationEfficiency):
    depends_on = ["s1_n_hits", "s1_area", "s2_area"]
    provides = ["anti_correlation_eff"]


@export
class FullEffSS(aptext.acceptance.efficiency_total.FullEffSS):
    depends_on = [
        "acc_s2_threshold",
        "acc_s1_recon_eff",
        "cut_acc_s1",
        "cut_acc_s2",
        "acc_fv_cut",
        "anti_correlation_eff",
    ]
    provides = ["eff"]

