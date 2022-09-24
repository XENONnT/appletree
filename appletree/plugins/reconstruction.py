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
class S1(Plugin):
    def __init__(self, par_manager : ParManager, map_manager : MapManager):
        super().__init__()
        
        self.par_names = []
        self.update_parameter(par_manager)
        
        self.map_names = ['s1_bias', 's1_smear']
        self.update_map(map_manager)
        
        self.input = ['num_s1_phd', 'num_s1_pe']
        self.output = ['s1']
        
    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, num_s1_phd, num_s1_pe):
        mean = interp.curve_interpolator(num_s1_phd, self.s1_bias.coordinate_system, self.s1_bias.map)
        std = interp.curve_interpolator(num_s1_phd, self.s1_smear.coordinate_system, self.s1_smear.map)
        key, bias = randgen.normal(key, mean, std)
        s1 = num_s1_pe * (1. + bias)
        return key, s1
    
    
@export
class S2(Plugin):
    def __init__(self, par_manager : ParManager, map_manager : MapManager):
        super().__init__()
        
        self.par_names = []
        self.update_parameter(par_manager)
        
        self.map_names = ['s2_bias', 's2_smear']
        self.update_map(map_manager)
        
        self.input = ['num_s2_pe']
        self.output = ['s2']
        
    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, num_s2_pe):
        mean = interp.curve_interpolator(num_s2_pe, self.s2_bias.coordinate_system, self.s2_bias.map)
        std = interp.curve_interpolator(num_s2_pe, self.s2_smear.coordinate_system, self.s2_smear.map)
        key, bias = randgen.normal(key, mean, std)
        s2 = num_s2_pe * (1. + bias)
        return key, s2
    
    
@export
class cS1(Plugin):
    def __init__(self, par_manager : ParManager, map_manager : MapManager):
        super().__init__()
        
        self.par_names = []
        self.update_parameter(par_manager)
        
        self.map_names = []
        self.update_map(map_manager)
        
        self.input = ['s1', 's1_correction']
        self.output = ['cs1']
    
    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, s1, s1_correction):
        cs1 = s1 / s1_correction
        return key, cs1
    
    
@export
class cS2(Plugin):
    def __init__(self, par_manager : ParManager, map_manager : MapManager):
        super().__init__()
        
        self.par_names = []
        self.update_parameter(par_manager)
        
        self.map_names = []
        self.update_map(map_manager)
        
        self.input = ['s2', 's2_correction', 'drift_survive_prob']
        self.output = ['cs2']
    
    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, s2, s2_correction, drift_survive_prob):
        cs2 = s2 / s2_correction / drift_survive_prob
        return key, cs2