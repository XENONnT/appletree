import jax.numpy as jnp
import numpy as np

from jax import jit
from functools import partial
from inspect import getsource

from appletree.ipm import ParManager
from appletree.imm import MapManager
from appletree.flex import randgen
from appletree import exporter

export, __all__ = exporter()

@export
class Plugin():
    def __init__(self):
        self.par_names = []
        self.par_values = np.array([])
        self.par_dict = {}
        
        self.map_names = []
        self.map_registration = {}
        
        self.input = []
        self.output = []
    
    def __call__(self, *args, **kwargs):
        return self.simulate(*args, **kwargs)
        
    def update_parameter(self, par_manager):
        if self.par_names == []:
            return
        
        check, missing = par_manager.check_parameter_exist(self.par_names, return_not_exist=True)
        assert check, "%s not found in par_manager!"%missing
        
        self.par_values = par_manager.get_parameter(self.par_names)
        self.par_dict = {key : val for key, val in zip(self.par_names, self.par_values)}
        
        for key, val in zip(self.par_names, self.par_values):
            self.__setattr__(key, val)
            
    def update_map(self, map_manager):
        if self.map_names == []:
            return
        
        registration = map_manager.registration
        
        for name in self.map_names:
            self.map_registration[name] = registration[name]
            self.__setattr__(name, map_manager.get_map(name))
    
    def simulate(self, *args, **kwargs):
        pass
    
    def get_doc(self):
        print("%s -> %s\n\nSource:\n"%(self.input, self.output) + getsource(self.simulate))
        
        
@export
class EnergySpectra(Plugin):
    def __init__(self, par_manager : ParManager, map_manager : MapManager, lower=0.01, upper=20.):
        super().__init__()
        
        self.par_names = []
        self.update_parameter(par_manager)
        
        self.map_names = []
        self.update_map(map_manager)
        
        self.input = ['batch_size']
        self.output = ['energy']
        
        self.lower = lower
        self.upper = upper
    
    @partial(jit, static_argnums=(0, 2))
    def simulate(self, key, batch_size):
        key, energy = randgen.uniform(key, self.lower, self.upper, shape=(batch_size, ))
        return key, energy
    
    
@export
class PositionSpectra(Plugin):
    def __init__(self, par_manager : ParManager, map_manager : MapManager, z_sim_min=-133.97, z_sim_max=-13.35, r_sim_max=60.):
        super().__init__()
        
        self.par_names = []
        self.update_parameter(par_manager)
        
        self.map_names = []
        self.update_map(map_manager)
        
        self.input = ['batch_size']
        self.output = ['x', 'y', 'z']
        
        self.z_lower = z_sim_min
        self.z_upper = z_sim_max
        self.r_upper = r_sim_max
    
    @partial(jit, static_argnums=(0, 2))
    def simulate(self, key, batch_size):
        key, z = randgen.uniform(key, self.z_lower, self.z_upper, shape=(batch_size, ))
        key, r2 = randgen.uniform(key, 0, self.r_upper**2, shape=(batch_size, ))
        key, theta = randgen.uniform(key, 0, 2*jnp.pi, shape=(batch_size, ))

        r = jnp.sqrt(r2)
        x = r * jnp.cos(theta)
        y = r * jnp.sin(theta)
        return key, x, y, z