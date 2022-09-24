import jax.numpy as jnp
from jax import jit
from functools import partial

from appletree.flex.plugin.common import Plugin
from appletree.flex import randgen
from appletree.flex import interp
from appletree.ipm import ParManager
from appletree.imm import MapManager
from appletree import exporter

export, __all__ = exporter(export_self=False)


@export
class S2Threshold(Plugin):
    def __init__(self, par_manager : ParManager, map_manager : MapManager):
        super().__init__()
        
        self.par_names = ['s2_threshold']
        self.update_parameter(par_manager)
        
        self.map_names = []
        self.update_map(map_manager)
        
        self.input = ['s2']
        self.output = ['acc_s2_threshold']
        
    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, s2):
        return key, jnp.where(s2 > self.s2_threshold, 1., 0)
    
    
@export
class S1ReconEff(Plugin):
    def __init__(self, par_manager : ParManager, map_manager : MapManager):
        super().__init__()
        
        self.par_names = []
        self.update_parameter(par_manager)
        
        self.map_names = ['s1_eff']
        self.update_map(map_manager)
        
        self.input = ['num_s1_phd']
        self.output = ['acc_s1_recon_eff']
        
    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, num_s1_phd):
        return key, interp.curve_interpolator(num_s1_phd, self.s1_eff.coordinate_system, self.s1_eff.map)