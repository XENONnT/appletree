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


class MapManager(object):
    def __init__(self):
        self._maps = {}

    def register_json_map(self, file_name, coord_type, map_name):
        if coord_type == 'point':
            map_cls = Map(file_name)
        elif coord_type == 'regbin':
            map_cls = MapRegBin(file_name)
        else:
            raise ValueError("map_type must be either 'point' or 'regbin'!")

        item = {
            'map' : map_cls,
            'bin_type' : coord_type,
            'file_name' : file_name
        }

        self._maps[map_name] = item

    @property
    def registration(self):
        registration = {map_name : map_dict['file_name'] for map_name, map_dict in self._maps.items()}
        return registration

    def get_map(self, map_name):
        return self._maps[map_name]['map']

    def check_map_exist(self, map_name):
        return map_name in self._maps
