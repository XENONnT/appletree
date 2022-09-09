import jax.numpy as jnp
import json

class Map(object):
    def __init__(self, file_name):
        with open(file_name, 'r') as file:
            self.data = json.load(file)
        self.coordinate_system = jnp.asarray(self.data['coordinate_system'], dtype=float)
        self.map = jnp.asarray(self.data['map'], dtype=float)
        self.coordinate_name = self.data['coordinate_name']

class MapRegBin(object):
    def __init__(self, file_name):
        with open(file_name, 'r') as file:
            self.data = json.load(file)
        self.coordinate_lowers = jnp.asarray(self.data['coordinate_lowers'], dtype=float)
        self.coordinate_uppers = jnp.asarray(self.data['coordinate_uppers'], dtype=float)
        self.map = jnp.asarray(self.data['map'], dtype=float)
        self.coordinate_name = self.data['coordinate_name']