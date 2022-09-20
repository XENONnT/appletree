import jax.numpy as jnp
import numpy as np

from jax import jit
from functools import partial
from inspect import getsource

from appletree.ipm import ParManager
from appletree.flex import randgen
from appletree.flex.utils import exporter

export, __all__ = exporter()

@export
class Plugin():
    def __init__(self):
        self.param_names = []
        self.param_values = np.array([])
        self.param_dict = {}
        
        self.input = []
        self.output = []
    
    def __call__(self, *args, **kwargs):
        return self.simulate(*args, **kwargs)
        
    def update_parameter(self, par_manager):
        check, missing = par_manager.check_parameter_exist(self.param_names, return_not_exist=True)
        assert check, "%s not found in par_manager!"%missing
        
        self.param_values = par_manager.get_parameter(self.param_names)
        self.param_dict = {key : val for key, val in zip(self.param_names, self.param_values)}
        
        for key, val in zip(self.param_names, self.param_values):
            self.__setattr__(key, val)
    
    def simulate(self, *args, **kwargs):
        pass
    
    def get_doc(self):
        print("%s -> %s\n\nSource:\n"%(self.input, self.output) + getsource(self.simulate))
        
        
@export
class EnergySpectra(Plugin):
    def __init__(self, par : ParManager, lower, upper):
        super().__init__()
        self.lower = lower
        self.upper = upper
        
        self.input = ['batch_size']
        self.output = ['energy']
    
    @partial(jit, static_argnums=(0, 2))
    def simulate(self, key, batch_size):
        key, energy = randgen.uniform(key, self.lower, self.upper, shape=(batch_size, ))
        return key, energy
    
    
@export
class PositionSpectra(Plugin):
    def __init__(self, par : ParManager, z_sim_min=-133.97, z_sim_max=-13.35, r_sim_max=60.):
        super().__init__()
        self.z_lower = z_sim_min
        self.z_upper = z_sim_max
        self.r_upper = r_sim_max
        
        self.input = ['batch_size']
        self.output = ['x', 'y', 'z']
    
    @partial(jit, static_argnums=(0, 2))
    def simulate(self, key, batch_size):
        key, z = randgen.uniform(key, self.z_lower, self.z_upper, shape=(batch_size, ))
        key, r2 = randgen.uniform(key, 0, self.r_upper**2, shape=(batch_size, ))
        key, theta = randgen.uniform(key, 0, 2*jnp.pi, shape=(batch_size, ))

        r = jnp.sqrt(r2)
        x = r * jnp.cos(theta)
        y = r * jnp.sin(theta)
        return key, x, y, z