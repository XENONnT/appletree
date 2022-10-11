from jax import numpy as jnp
from jax import jit
from functools import partial

import appletree
from appletree import interpolation
from appletree.plugin import Plugin
from appletree.config import Map
from appletree.utils import exporter

export, __all__ = exporter(export_self=False)


@export
class S2Threshold(Plugin):
    depends_on = ['s2']
    provides = ['acc_s2_threshold']
    parameters = ('s2_threshold',)

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, s2):
        return key, jnp.where(s2 > parameters['s2_threshold'], 1., 0)


@export
@appletree.takes_config(
    Map(name='s1_eff',
        default='3fold_recon_eff.json',
        help='S1 light collation efficiency correction'),
)
class S1ReconEff(Plugin):
    depends_on = ['num_s1_phd']
    provides = ['acc_s1_recon_eff']

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, num_s1_phd):
        acc_s1_recon_eff = interpolation.curve_interpolator(num_s1_phd,
            self.s1_eff.coordinate_system, self.s1_eff.map)
        return key, acc_s1_recon_eff


@export
@appletree.takes_config(
    Map(name='s1_cut_acc',
        default='s1_cut_acc.json',
        help='S1 cut acceptance'),
)
class S1CutAccept(Plugin):
    depends_on = ['s1']
    provides = ['cut_acc_s1']

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, s1):
        cut_acc_s1 = interpolation.curve_interpolator(s1,
            self.s1_cut_acc.coordinate_system, self.s1_cut_acc.map)
        return key, cut_acc_s1


@export
@appletree.takes_config(
    Map(name='s2_cut_acc',
        default='s2_cut_acc.json',
        help='S2 cut acceptance'),
)
class S2CutAccept(Plugin):
    depends_on = ['s2']
    provides = ['cut_acc_s2']

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, s2):
        cut_acc_s2 = interpolation.curve_interpolator(s2,
            self.s2_cut_acc.coordinate_system, self.s2_cut_acc.map)
        return key, cut_acc_s2


@export
class Eff(Plugin):
    depends_on = ['acc_s2_threshold', 'acc_s1_recon_eff', 'cut_acc_s1', 'cut_acc_s2']
    provides = ['eff']

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, acc_s2_threshold, acc_s1_recon_eff, cut_acc_s1, cut_acc_s2):
        eff = acc_s2_threshold * acc_s1_recon_eff * cut_acc_s1 * cut_acc_s2
        return key, eff
