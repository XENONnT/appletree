from appletree.share import cached_functions
from functools import partial
from jax import random, jit
from appletree.hist import *
import jax.numpy as jnp
import pandas as pd

class Component(object):
    def __init__(self,
                 config:dict,
                 *args,
                 **kwargs):
        self.name = config['name']
        self.axis = config['axis']
        self.bins = config['bins']
        self.bins_type = config['bins_type']
        self.norm = config['norm']
        self.norm_type = config['norm_type']
        self.sim = config['sim']
        self.sim_type = config['sim_type']
        self.batch_size = config['batch_size']
        
        self.dim = len(self.axis)
        self.sanity_check()
        
        if self.sim_type == 'context':
            self.init_context(*args, **kwargs)
        elif self.sim_type == 'bootstrap':
            self.init_bootstrap()
                
    def sanity_check(self):
        assert self.bins_type in ['meshgrid', 'irreg'], f"bins_type '{self.bins_type}' not supported!"
        assert self.norm_type in ['on_pdf', 'on_sim'], f"norm_type '{self.norm_type}' not supported!"
        assert self.sim_type in ['context', 'bootstrap'], f"sim_type '{self.sim_type}' not supported!"
        assert len(self.axis) == len(self.bins), "length of axis must be equal to length of bins!"
        
        if self.bins_type == 'irreg':
            if self.dim == 1:
                self.bins_type = 'meshgrid'
            elif self.dim != 2:
                raise ValueError('irreg is only supported for 1 or 2 dimension!')
                
        if self.sim_type == 'bootstrap':
            assert isinstance(self.sim, str), "if sim_type is bootstrap, sim must be a path to the file!"
        
    def init_context(self, *args, **kwargs):
        # here we assume func_name won't confict with others. Should be checked outside Component
        self.sim_cls = self.sim(*args, **kwargs)
        self.sim_cls.deduce(
            data_names = self.axis + ['eff'],
            func_name = self.name,
        )
        self.sim_cls.compile()
        
        self.simulate = lambda key, parameters: cached_functions[self.name](key, self.batch_size, parameters)
    
    def init_bootstrap(self):
        fmt = self.sim.split('.')[-1]
        
        if fmt == 'csv':
            self.sim_dset = pd.read_csv(self.sim)[self.axis].to_numpy()
        elif fmt == 'pkl':
            self.sim_dset = pd.read_pickel(self.sim)[self.axis].to_numpy()
        else:
            raise ValueError(f'unsupported format {fmt}!')
            
        self.sim_dset = jnp.array(self.sim_dset)
        
        @partial(jit, static_argnums=(1, ))
        def _bootstrap(key, batch_size, parameters, dset):
            key, seed = random.split(key)
            ind = random.randint(seed, (batch_size, ), 0, len(dset))
            eff = jnp.ones(batch_size)
            return key, jnp.vstack([dset[ind].T, eff])
            
        self.simulate = lambda key, parameters: _bootstrap(key, self.batch_size, parameters, self.sim_dset)
        
    def simulate_hist(self, key, parameters):
        key, result = self.simulate(key, parameters)
        
        mc = jnp.asarray(result[:-1])
        eff = result[-1]
        
        if self.bins_type == 'meshgrid':
            hist = make_hist_mesh_grid(mc.T, bins=self.bins, weights=eff)
        elif self.bins_type == 'irreg':
            hist = make_hist_irreg_bin_2d(mc.T, self.bins[0], self.bins[1], weights=eff)
        else:
            raise ValueError(f'unsupported bins_type {self.bins_type}')
            
        hist = hist + 1. # as uncertainty to prevent likelihood blow up
            
        if self.norm_type == 'on_pdf':
            hist = hist / jnp.sum(hist) * self.norm
        elif self.norm_type == 'on_sim':
            hist = hist / self.batch_size * self.norm
            
        return key, hist
