import jax.numpy as jnp
from jax import jit
from functools import partial

from appletree.flex.plugin.common import Plugin
from appletree.flex import randgen
from appletree.flex import interp
from appletree.imm import MapRegBin, Map
from appletree.ipm import ParManager

class S2Threshold(Plugin):
    def __init__(self, par : ParManager):
        super().__init__()
        self.param_names = ['s2_threshold']
        self.update_parameter(par)
        
        self.input = ['s2']
        self.output = ['acc_s2_threshold']
        
    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, s2):
        return key, jnp.where(s2 > self.s2_threshold, 1., 0)
    
    
class S1ReconEff(Plugin):
    def __init__(self, par : ParManager, s1_eff : Map):
        super().__init__()
        
        self.input = ['num_s1_phd']
        self.output = ['acc_s1_recon_eff']
        
        self.map_coordinate = s1_eff.coordinate_system
        self.map = s1_eff.map
        
    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, num_s1_phd):
        return key, interp.curve_interpolator(num_s1_phd, self.map_coordinate, self.map)