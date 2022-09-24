import typing as ty
import json
from enum import IntEnum
from immutabledict import immutabledict

import jax.numpy as jnp

from appletree import exporter

export, __all__ = exporter()

OMITTED = '<OMITTED>'
__all__ += 'OMITTED InvalidConfiguration'.split()

@export
def takes_map(*maps):
    """
    Decorator for plugin classes, to specify which maps it takes.
    :param maps: Map instances of maps this plugin takes.
    """
    def wrapped(plugin_class):
        result = {}
        for map in maps:
            if not isinstance(map, Map):
                raise RuntimeError("Specify config options by Map objects")
            map.taken_by = plugin_class.__name__
            result[map.name] = map

        if (hasattr(plugin_class, 'takes_map')
                and len(plugin_class.takes_map)):
            # Already have some maps set, e.g. because of subclassing
            # where both child and parent have a takes_map decorator
            for map in result.values():
                if map.name in plugin_class.takes_map:
                    raise RuntimeError(
                        f'Attempt to specify map {map.name} twice')
            plugin_class.takes_map = immutabledict({
                **plugin_class.takes_map, **result})
        else:
            plugin_class.takes_map = immutabledict(result)

        for map in plugin_class.takes_map:
            setattr(plugin_class, map.name, map)
        return plugin_class

    return wrapped

@export
class MapType(IntEnum):
    """
    Identifies what type of map
    """
    # Map has only discrete points
    POINT = 0
    # Map has regular binning, like a meshgrid
    REGBIN = 1


class Map(object):
    taken_by: str

    def __init__(self,
                 name: str,
                 coord_type: ty.Union[type, tuple, list] = OMITTED,
                 file_name: ty.Union[type, tuple, list] = OMITTED,
                 default: ty.Any = OMITTED,
                 help: str = ''):
        self.name = name
        self.coord_type = coord_type
        self.file_name = file_name
        self.default = default
        self.help = help

        self.build(self.coord_type, self.file_name)

    def build(self, type, file_name):
        if type == 'point':
            self.type = MapType.POINT
            self.build_point(file_name)
        elif type == 'regbin':
            self.type = MapType.REGBIN
            self.build_regbin(file_name)
        else:
            raise ValueError("map_type must be either 'point' or 'regbin'!")

    def build_point(self, file_name):
        with open(file_name, 'r') as file:
            self.data = json.load(file)
        self.coordinate_name = self.data['coordinate_name']
        self.coordinate_system = jnp.asarray(self.data['coordinate_system'], dtype=float)
        self.map = jnp.asarray(self.data['map'], dtype=float)

    def build_regbin(self, file_name):
        with open(file_name, 'r') as file:
            self.data = json.load(file)
        self.coordinate_name = self.data['coordinate_name']
        self.coordinate_lowers = jnp.asarray(self.data['coordinate_lowers'], dtype=float)
        self.coordinate_uppers = jnp.asarray(self.data['coordinate_uppers'], dtype=float)
        self.map = jnp.asarray(self.data['map'], dtype=float)
