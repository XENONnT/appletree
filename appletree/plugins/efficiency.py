import jax.numpy as jnp
from jax import jit
from functools import partial

import appletree
from appletree.plugin import Plugin
from appletree import interpolation
from appletree.mapping import Mapping
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
@appletree.takes_map(
    Mapping(name='s1_eff',
        coord_type='point',
        file_name='3fold_recon_eff.json',
        doc='S1 light collation efficiency correction')
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
class Eff(Plugin):
    depends_on = ['acc_s2_threshold', 'acc_s1_recon_eff']
    provides = ['eff']

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, acc_s2_threshold, acc_s1_recon_eff):
        eff = acc_s2_threshold * acc_s1_recon_eff
        return key, eff
