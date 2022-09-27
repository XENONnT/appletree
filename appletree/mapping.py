import os
import inspect
import typing as ty
import json
from enum import IntEnum

from immutabledict import immutabledict
import jax.numpy as jnp

from appletree.utils import exporter
from appletree.share import MAPPATH

export, __all__ = exporter()

OMITTED = '<OMITTED>'

__all__ += 'OMITTED MAPPATH'.split()

@export
def takes_map(*maps):
    """
    Decorator for plugin classes, to specify which maps it takes.
    :param maps: Mapping instances of maps this plugin takes.
    """
    def wrapped(plugin_class):
        result = {}
        for mapping in maps:
            if not isinstance(mapping, Mapping):
                raise RuntimeError("Specify config options by Mapping objects")
            mapping.taken_by = plugin_class.__name__
            result[mapping.name] = mapping

        if (hasattr(plugin_class, 'takes_map')
                and len(plugin_class.takes_map)):
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
    """
    Identifies what type of mapping
    """
    # Mapping has only discrete points
    POINT = 0
    # Mapping has regular binning, like a meshgrid
    REGBIN = 1


class Mapping(object):
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

        # We don't load maps when initializing. We load maps when initialize plugins.
        # self.build(self.coord_type, self.file_name)

    def build(self, type, file_name):
        file_path = os.path.join(MAPPATH, file_name)
        if type == 'point':
            self.type = MapType.POINT
            self.build_point(file_path)
        elif type == 'regbin':
            self.type = MapType.REGBIN
            self.build_regbin(file_path)
        else:
            raise ValueError("map_type must be either 'point' or 'regbin'!")

    def build_point(self, file_path):
        with open(file_path, 'r') as file:
            self.data = json.load(file)
        self.coordinate_name = self.data['coordinate_name']
        self.coordinate_system = jnp.asarray(self.data['coordinate_system'], dtype=float)
        self.map = jnp.asarray(self.data['map'], dtype=float)

    def build_regbin(self, file_path):
        with open(file_path, 'r') as file:
            self.data = json.load(file)
        self.coordinate_name = self.data['coordinate_name']
        self.coordinate_lowers = jnp.asarray(self.data['coordinate_lowers'], dtype=float)
        self.coordinate_uppers = jnp.asarray(self.data['coordinate_uppers'], dtype=float)
        self.map = jnp.asarray(self.data['map'], dtype=float)
