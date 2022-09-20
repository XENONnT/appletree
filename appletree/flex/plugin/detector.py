import jax.numpy as jnp
from jax import jit
from functools import partial

from appletree.flex.plugin.common import Plugin
from appletree.flex import randgen
from appletree.flex import interp
from appletree.imm import MapRegBin, Map
from appletree.ipm import ParManager

class S1Correction(Plugin):
    def __init__(self, par : ParManager, s1_lce_map : MapRegBin):
        super().__init__()
        
        self.input = ['x', 'y', 'z']
        self.output = ['s1_correction']
        
        self.map_lowers = s1_lce_map.coordinate_lowers
        self.map_uppers = s1_lce_map.coordinate_uppers
        self.map = s1_lce_map.map
        
    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, x, y, z):
        pos = jnp.stack([x, y, z]).T
        s1_correction = interp.map_interpolator_regular_binning_3d(
            pos,
            self.map_lowers,
            self.map_uppers,
            self.map
        )
        return key, s1_correction
    
    
class S2Correction(Plugin):
    def __init__(self, par : ParManager, s2_lce_map : MapRegBin):
        super().__init__()
        
        self.input = ['x', 'y']
        self.output = ['s2_correction']
        
        self.map_lowers = s2_lce_map.coordinate_lowers
        self.map_uppers = s2_lce_map.coordinate_uppers
        self.map = s2_lce_map.map
        
    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, x, y):
        pos = jnp.stack([x, y]).T
        s2_correction = interp.map_interpolator_regular_binning_2d(
            pos,
            self.map_lowers,
            self.map_uppers,
            self.map
        )
        return key, s2_correction
    
    
class PhotonDetection(Plugin):
    def __init__(self, par : ParManager):
        super().__init__()
        self.param_names = ['g1', 'p_dpe']
        self.update_parameter(par)
        
        self.input = ['num_photon', 's1_correction']
        self.output = ['num_s1_phd']
    
    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, num_photon, s1_correction):
        g1_true_no_dpe = jnp.clip(self.g1 * s1_correction / (1. + self.p_dpe), 0, 1.)
        key, num_s1_phd = randgen.binomial(key, g1_true_no_dpe, num_photon)
        return key, num_s1_phd
    
    
class S1PE(Plugin):
    def __init__(self, par : ParManager):
        super().__init__()
        self.param_names = ['p_dpe']
        self.update_parameter(par)
        
        self.input = ['num_s1_phd']
        self.output = ['num_s1_pe']
    
    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, num_s1_phd):
        key, num_s1_dpe = randgen.binomial(key, self.p_dpe, num_s1_phd)
        num_s1_pe = num_s1_dpe + num_s1_phd
        return key, num_s1_pe
    
    
class DriftLoss(Plugin):
    def __init__(self, par : ParManager, elife_map : Map):
        super().__init__()
        self.param_names = ['drift_velocity']
        self.update_parameter(par)
        
        self.input = ['z']
        self.output = ['drift_survive_prob']
        
        self.map_coordinate = elife_map.coordinate_system
        self.map = elife_map.map
    
    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, z):
        key, p = randgen.uniform(key, 0, 1., shape=jnp.shape(z))
        lifetime = interp.curve_interpolator(p, self.map_coordinate, self.map)
        drift_survive_prob = jnp.exp(- jnp.abs(z) / self.drift_velocity / lifetime)
        return key, drift_survive_prob
    
    
class ElectronDrifted(Plugin):
    def __init__(self, par : ParManager):
        super().__init__()
        
        self.input = ['num_electron', 'drift_survive_prob']
        self.output = ['num_electron_drifted']
    
    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, num_electron, drift_survive_prob):
        key, num_electron_drifted = randgen.binomial(key, drift_survive_prob, num_electron)
        return key, num_electron_drifted
        
        
class S2PE(Plugin):
    def __init__(self, par : ParManager):
        super().__init__()
        self.param_names = ['g2', 'gas_gain']
        self.update_parameter(par)
        
        self.input = ['num_electron_drifted', 's2_correction']
        self.output = ['num_s2_pe']
    
    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, num_electron_drifted, s2_correction):
        extraction_eff = self.g2 / self.gas_gain
        g2_true = self.g2 * s2_correction
        gas_gain_true = g2_true / extraction_eff

        key, num_electron_extracted = randgen.binomial(key, extraction_eff, num_electron_drifted)

        mean_s2_pe = num_electron_extracted * gas_gain_true
        key, num_s2_pe = randgen.truncate_normal(key, mean_s2_pe, jnp.sqrt(mean_s2_pe), vmin=0)

        return key, num_s2_pe
        