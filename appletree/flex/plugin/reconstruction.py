import jax.numpy as jnp
from jax import jit
from functools import partial

from appletree.flex.plugin.common import Plugin
from appletree.flex import randgen
from appletree.flex import interp
from appletree.imm import MapRegBin, Map
from appletree.ipm import ParManager
from appletree import exporter

export, __all__ = exporter(export_self=False)


@export
class S1(Plugin):
    def __init__(self, par : ParManager, s1_bias : Map, s1_smear : Map):
        super().__init__()
        
        self.input = ['num_s1_phd', 'num_s1_pe']
        self.output = ['s1']
        
        self.map_coordinate_bias = s1_bias.coordinate_system
        self.map_coordinate_smear = s1_smear.coordinate_system
        self.map_bias = s1_bias.map
        self.map_smear = s1_smear.map
        
    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, num_s1_phd, num_s1_pe):
        mean = interp.curve_interpolator(num_s1_phd, self.map_coordinate_bias, self.map_bias)
        std = interp.curve_interpolator(num_s1_phd, self.map_coordinate_smear, self.map_smear)
        key, bias = randgen.normal(key, mean, std)
        s1 = num_s1_pe * (1. + bias)
        return key, s1
    
    
@export
class S2(Plugin):
    def __init__(self, par : ParManager, s2_bias : Map, s2_smear : Map):
        super().__init__()
        
        self.input = ['num_s2_pe']
        self.output = ['s2']
        
        self.map_coordinate_bias = s2_bias.coordinate_system
        self.map_coordinate_smear = s2_smear.coordinate_system
        self.map_bias = s2_bias.map
        self.map_smear = s2_smear.map
        
    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, num_s2_pe):
        mean = interp.curve_interpolator(num_s2_pe, self.map_coordinate_bias, self.map_bias)
        std = interp.curve_interpolator(num_s2_pe, self.map_coordinate_smear, self.map_smear)
        key, bias = randgen.normal(key, mean, std)
        s2 = num_s2_pe * (1. + bias)
        return key, s2
    
    
@export
class cS1(Plugin):
    def __init__(self, par : ParManager):
        super().__init__()
        
        self.input = ['s1', 's1_correction']
        self.output = ['cs1']
    
    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, s1, s1_correction):
        cs1 = s1 / s1_correction
        return key, cs1
    
    
@export
class cS2(Plugin):
    def __init__(self, par : ParManager):
        super().__init__()
        
        self.input = ['s2', 's2_correction', 'drift_survive_prob']
        self.output = ['cs2']
    
    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, s2, s2_correction, drift_survive_prob):
        cs2 = s2 / s2_correction / drift_survive_prob
        return key, cs2