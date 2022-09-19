import numpy as np

from inspect import getsource
from utils import exporter

from appletree.ipm import ParManager
from appletree.flex import randgen

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