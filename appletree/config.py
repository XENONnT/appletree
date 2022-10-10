import os
import typing as ty
import logging

from immutabledict import immutabledict
from jax import numpy as jnp

import straxen
from appletree.utils import exporter, load_json
from appletree.share import _cached_configs

export, __all__ = exporter()

OMITTED = '<OMITTED>'

__all__ += 'OMITTED'.split()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('appletree.config')
log.setLevel('WARNING')

NT_AUX_INSTALLED = False
try:
    import ntauxfiles
    NT_AUX_INSTALLED = True
except (ModuleNotFoundError, ImportError):
    pass


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
        else:
            self.value = _cached_configs[self.name]


@export
class Map(Config):
    """Map is a special config which takes input file"""

    def build(self):
        """Cache the map to jnp.array"""
        print(f'Building {self.name} map')

        if self.name in _cached_configs:
            file_path = _cached_configs[self.name]
        else:
            file_path = get_file_path(_cached_configs['url_base'], self.get_default())
            _cached_configs.update({self.name: file_path})

        data = load_json(file_path)

        if data['coord_type'] == 'point':
            self.build_point(data)
        elif data['coord_type'] == 'regbin':
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


# Copied from https://github.com/XENONnT/WFSim/blob/master/wfsim/load_resource.py
@export
def get_file_path(base, fname):
    """Find the full path to the resource file
    Try 4 methods in the following order
    1. The base is not url, return base + name
    2. If ntauxfiles (straxauxfiles) is installed, return will be package dir + name
        pip install won't work, try python setup.py in the packages
    3. Download the latest version using straxen mongo downloader from database,
        return the cached file path + md5
    4. Download using straxen get_resource from the url (github raw)
        simply return base + name
        Be careful with the forth options, straxen creates
        cache files that might not be updated with the latest github commit.
    """
    if not fname:
        log.warning(f"A file has value False, assuming this is intentional.")
        return

    if fname.startswith('/'):
        log.warning(f"Using local file {fname} for a resource. "
                    f"Do not set this as a default or TravisCI tests will break")
        return fname
    
    if base.startswith('/'):
        log.warning(f"Using local folder {base} for all resources. "
                    f"Do not set this as a default or TravisCI tests will break")
        return os.path.join(base, fname)

    if NT_AUX_INSTALLED:
        # You might want to use this, for example if you are a developer
        if fname in ntauxfiles.list_private_files():
            log.warning(f"Using the private repo to load {fname} locally")
            fpath = ntauxfiles.get_abspath(fname)
            log.info(f"Loading {fname} is successfully from {fpath}")
            return fpath

    try:
        # https://straxen.readthedocs.io/en/latest/config_storage.html
        # downloading-xenonnt-files-from-the-database  # noqa

        # we need to add the straxen.MongoDownloader() in this
        # try: except NameError: logic because the NameError
        # gets raised if we don't have access to utilix.
        downloader = straxen.MongoDownloader()
        # FileNotFoundError, ValueErrors can be raised if we
        # cannot load the requested config
        fpath = downloader.download_single(fname)
        log.warning(f"Loading {fname} from mongo downloader to {fpath}")
        return fname  # Keep the name and let get_resource do its thing

    except (FileNotFoundError, ValueError, NameError, AttributeError):
        log.info(f"Mongo downloader not possible or does not have {fname}")

    # We cannot download the file from the database. We need to
    # try to get a placeholder file from a URL.
    furl = os.path.join(base, fname)
    log.warning(f'{fname} did not download, trying download from {base}')
    return furl
