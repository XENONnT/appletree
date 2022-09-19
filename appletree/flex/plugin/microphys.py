import jax.numpy as jnp
from jax import jit
from functools import partial

from appletree.flex import randgen
from appletree.flex import Plugin
from appletree.ipm import ParManager
    
class Quenching(Plugin):
    def __init__(self, par : ParManager):
        super().__init__()
        self.param_names = ['w', 'fano']
        self.update_parameter(par)
        
        self.input = ['energy']
        self.output = ['num_quanta']
        
    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, energy):
        num_quanta_mean = energy / self.w
        num_quanta_std = jnp.sqrt(num_quanta_mean * self.fano)
        key, num_quanta = randgen.truncate_normal(key, num_quanta_mean, num_quanta_std, vmin=0)
        return key, num_quanta.round().astype(int)
    
    
class Ionization(Plugin):
    def __init__(self, par : ParManager):
        super().__init__()
        self.param_names = ['nex_ni_ratio']
        self.update_parameter(par)
        
        self.input = ['num_quanta']
        self.output = ['num_ion']
        
    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, num_quanta):
        p_ion = 1. / (1. + self.nex_ni_ratio)
        key, num_ion = randgen.binomial(key, p_ion, num_quanta)
        return key, num_ion
    
    
class mTI(Plugin):
    def __init__(self, par : ParManager):
        super().__init__()
        self.param_names = ['w', 'nex_ni_ratio', 'py0', 'py1', 'py2', 'py3', 'py4', 'field']
        self.update_parameter(par)
        
        self.input = ['energy']
        self.output = ['recomb_mean']
        
    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, energy):
        ni = energy / self.w / (1. + self.nex_ni_ratio)
        ti = ni * self.py0 * jnp.exp(- energy / self.py1) * self.field**self.py2 / 4.
        fd = 1. / (1. + jnp.exp(- (energy - self.py3) / self.py4))
        r = jnp.where(
            ti < 1e-2, 
            ti / 2. - ti * ti / 3., 
            1. - jnp.log(1. + ti) / ti
        )
        return key, r * fd
    
    
class RecombFluct(Plugin):
    def __init__(self, par : ParManager):
        super().__init__()
        self.param_names = ['rf0', 'rf1']
        self.update_parameter(par)
        
        self.input = ['energy']
        self.output = ['recomb_std']
        
    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, energy):
        return key, jnp.clip(self.rf0 * (1. - jnp.exp(- energy / self.rf1)), 0, 1.)
    
    
class TrueRecomb(Plugin):
    def __init__(self, par : ParManager):
        super().__init__()
        
        self.input = ['recomb_mean', 'recomb_std']
        self.output = ['recomb']
        
    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, recomb_mean, recomb_std):
        key, recomb = randgen.truncate_normal(key, recomb_mean, recomb_std, vmin=0., vmax=1.)
        return key, recomb
    
    
class Recombination(Plugin):
    def __init__(self, par : ParManager):
        super().__init__()
        
        self.input = ['num_quanta', 'num_ion', 'recomb']
        self.output = ['num_photon', 'num_electron']
        
    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, num_quanta, num_ion, recomb):
        p_not_recomb = 1. - recomb
        key, num_electron = randgen.binomial(key, p_not_recomb, num_ion)
        num_photon = num_quanta - num_electron
        
        return key, num_photon, num_electron