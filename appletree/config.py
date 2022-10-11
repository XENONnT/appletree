import typing as ty

from immutabledict import immutabledict
from jax import numpy as jnp

from appletree.share import _cached_configs
from appletree.utils import exporter, load_json, get_file_path

export, __all__ = exporter()

OMITTED = '<OMITTED>'

__all__ += 'OMITTED'.split()


@export
def takes_config(*configs):
    """Decorator for plugin classes, to specify which configs it takes.
    :param configs: Config instances of configs this plugin takes.
    """

    def wrapped(plugin_class):
        """
        :param plugin_class: plugin needs configuration
        """
        result = {}
        for config in configs:
            if not isinstance(config, Config):
                raise RuntimeError("Specify config options by Config objects")
            config.taken_by = plugin_class.__name__
            result[config.name] = config

        if (hasattr(plugin_class, 'takes_config') and len(plugin_class.takes_config)):
            # Already have some configs set, e.g. because of subclassing
            # where both child and parent have a takes_config decorator
            for config in result.values():
                if config.name in plugin_class.takes_config:
                    raise RuntimeError(
                        f'Attempt to specify config {config.name} twice')
            plugin_class.takes_config = immutabledict({
                **plugin_class.takes_config, **result})
        else:
            plugin_class.takes_config = immutabledict(result)

        for config in plugin_class.takes_config.values():
            setattr(plugin_class, config.name, config)
        return plugin_class

    return wrapped


@export
class Config():
    """Configuration option taken by a appletree plugin"""

    def __init__(self,
                 name: str,
                 type: ty.Union[type, tuple, list] = OMITTED,
                 default: ty.Any = OMITTED,
                 help: str = ''):
        """Initialization.
        :param name: name of the map
        :param type: Excepted type of the option's value.
        :param default: Default value the option takes.
        :param help: description of the map
        """
        self.name = name
        self.type = type
        self.default = default
        self.help = help

    def get_default(self):
        """Get default value of configuration"""
        if self.default is not OMITTED:
            return self.default

        raise ValueError(f"Missing option {self.name} "
                         f"required by {self.taken_by}")

    def build(self):
        """Build configuration, set attributes to Config instance"""
        raise NotImplementedError


@export
class Constant(Config):
    """Map is a special config which takes only certain value"""

    value = None

    def build(self):
        """Set value of Constant"""
        if not self.name in _cached_configs:
            _cached_configs.update({self.name: self.get_default()})

        self.value = _cached_configs[self.name]


@export
class Map(Config):
    """Map is a special config which takes input file"""

    def build(self):
        """Cache the map to jnp.array"""

        if self.name in _cached_configs:
            file_path = _cached_configs[self.name]
        else:
            file_path = get_file_path(self.get_default())
            _cached_configs.update({self.name: file_path})

        data = load_json(file_path)

        if data['coordinate_type'] == 'point':
            self.build_point(data)
        elif data['coordinate_type'] == 'regbin':
            self.build_regbin(data)
        else:
            raise ValueError("map_type must be either 'point' or 'regbin'!")

    def build_point(self, data):
        """Cache the map to jnp.array if bins_type is point"""

        self.coordinate_name = data['coordinate_name']
        self.coordinate_system = jnp.asarray(data['coordinate_system'], dtype=float)
        self.map = jnp.asarray(data['map'], dtype=float)

    def build_regbin(self, data):
        """Cache the map to jnp.array if bins_type is regbin"""
        
        self.coordinate_name = data['coordinate_name']
        self.coordinate_lowers = jnp.asarray(data['coordinate_lowers'], dtype=float)
        self.coordinate_uppers = jnp.asarray(data['coordinate_uppers'], dtype=float)
        self.map = jnp.asarray(data['map'], dtype=float)
