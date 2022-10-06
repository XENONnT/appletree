import os
import typing as ty
import json
from enum import IntEnum

from immutabledict import immutabledict
from jax import numpy as jnp

from appletree.utils import exporter
from appletree.share import MAPPATH

export, __all__ = exporter()

OMITTED = '<OMITTED>'

__all__ += 'OMITTED'.split()


@export
def takes_map(*maps):
    """Decorator for plugin classes, to specify which maps it takes.
    :param maps: Mapping instances of maps this plugin takes.
    """

    def wrapped(plugin_class):
        result = {}
        for mapping in maps:
            if not isinstance(mapping, Mapping):
                raise RuntimeError("Specify config options by Mapping objects")
            mapping.taken_by = plugin_class.__name__
            result[mapping.name] = mapping

        if (hasattr(plugin_class, 'takes_map') and len(plugin_class.takes_map)):
            # Already have some maps set, e.g. because of subclassing
            # where both child and parent have a takes_map decorator
            for mapping in result.values():
                if mapping.name in plugin_class.takes_map:
                    raise RuntimeError(
                        f'Attempt to specify mapping {mapping.name} twice')
            plugin_class.takes_map = immutabledict({
                **plugin_class.takes_map, **result})
        else:
            plugin_class.takes_map = immutabledict(result)

        for mapping in plugin_class.takes_map.values():
            setattr(plugin_class, mapping.name, mapping)
        return plugin_class

    return wrapped


@export
class MapType(IntEnum):
    """Identifies what type of mapping"""

    # Mapping has only discrete points
    POINT = 0
    # Mapping has regular binning, like a meshgrid
    REGBIN = 1


class Mapping(object):
    """Wrapper of an input map."""

    taken_by: str

    def __init__(self,
                 name: str,
                 coord_type: ty.Union[type, tuple, list] = OMITTED,
                 file_name: ty.Union[type, tuple, list] = OMITTED,
                 default: ty.Any = OMITTED,
                 help: str = ''):
        """Initialization.
        :param name: name of the map
        :param coord_type: how the coordination is provided. Can be point or regbin
        :param file_name: file name of the map
        :param help: description of the map
        """
        self.name = name
        self.coord_type = coord_type
        self.file_name = file_name
        self.default = default
        self.help = help

    def build(self, bins_type, file_name):
        """Cache the map to jnp.array"""
        file_path = os.path.join(MAPPATH, file_name)
        if bins_type == 'point':
            self.type = MapType.POINT
            self.build_point(file_path)
        elif bins_type == 'regbin':
            self.type = MapType.REGBIN
            self.build_regbin(file_path)
        else:
            raise ValueError("map_type must be either 'point' or 'regbin'!")

    def build_point(self, file_path):
        """Cache the map to jnp.array if bins_type is point"""
        with open(file_path, 'r') as file:
            self.data = json.load(file)
        self.coordinate_name = self.data['coordinate_name']
        self.coordinate_system = jnp.asarray(self.data['coordinate_system'], dtype=float)
        self.map = jnp.asarray(self.data['map'], dtype=float)

    def build_regbin(self, file_path):
        """Cache the map to jnp.array if bins_type is regbin"""
        with open(file_path, 'r') as file:
            self.data = json.load(file)
        self.coordinate_name = self.data['coordinate_name']
        self.coordinate_lowers = jnp.asarray(self.data['coordinate_lowers'], dtype=float)
        self.coordinate_uppers = jnp.asarray(self.data['coordinate_uppers'], dtype=float)
        self.map = jnp.asarray(self.data['map'], dtype=float)
