import typing as ty

from immutabledict import immutabledict
from jax import numpy as jnp
from warnings import warn

from appletree.share import _cached_configs
from appletree.utils import exporter, load_json, get_file_path, integrate_midpoint, cum_integrate_midpoint
from appletree import interpolation
from appletree.interpolation import FLOAT_POS_MIN, FLOAT_POS_MAX

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

        # Should set the configurations as the attributes of Plugin
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

        # Sanity check
        if isinstance(self.default, dict):
            raise ValueError(
                f"Do not set {self.name}'s default value as dict!",
            )

    def get_default(self):
        """Get default value of configuration"""
        if self.default is not OMITTED:
            return self.default

        raise ValueError(f"Missing option {self.name} "
                         f"required by {self.taken_by}")

    def build(self, llh_name: str = None):
        """Build configuration, set attributes to Config instance"""
        raise NotImplementedError


@export
class Constant(Config):
    """Constant is a special config which takes only certain value"""

    value = None

    def build(self, llh_name: str = None):
        """Set value of Constant"""
        if self.name in _cached_configs:
            value = _cached_configs[self.name]
        else:
            value = self.get_default()
            # Update values to sharing dictionary
            _cached_configs[self.name] = value

        if isinstance(value, dict):
            try:
                self.value = value[llh_name]
            except KeyError:
                mesg = f'You specified {self.name} as a dictionary. '
                mesg += f'The key of it should be the name of one '
                mesg += f'of the likelihood, '
                mesg += f'but it is {llh_name}.'
                raise ValueError(mesg)
        else:
            self.value = value


@export
class Map(Config):
    """
    Map is a special config that takes input files.
    The method `apply` is dynamically assigned.
    When using points, the `apply` will be `map_point`, 
    while using regular binning, the `apply` will be `map_regbin`.
    When using log-binning, we will first convert the positions to log space.
    """

    def build(self, llh_name: str = None):
        """Cache the map to jnp.array"""

        if self.name in _cached_configs:
            file_path = _cached_configs[self.name]
        else:
            file_path = get_file_path(self.get_default())
            # Update values to sharing dictionary
            _cached_configs[self.name] = file_path

        if isinstance(file_path, dict):
            try:
                self.file_path = file_path[llh_name]
            except KeyError:
                mesg = f'You specified {self.name} as a dictionary. '
                mesg += f'The key of it should be the name of one '
                mesg += f'of the likelihood, '
                mesg += f'but it is {llh_name}.'
                raise ValueError(mesg)
        else:
            self.file_path = file_path

        data = load_json(self.file_path)

        coordinate_type = data['coordinate_type']
        if coordinate_type == 'point' or coordinate_type == 'log_point':
            self.build_point(data)
        elif coordinate_type == 'regbin' or coordinate_type == 'log_regbin':
            self.build_regbin(data)
        else:
            raise ValueError("map_type must be either 'point' or 'regbin'!")

    def build_point(self, data):
        """Cache the map to jnp.array if bins_type is point"""
        if data['coordinate_name'] == 'pdf' or data['coordinate_name'] == 'cdf':
            if data['coordinate_type'] == 'log_point':
                raise ValueError(
                    f'It is not a good idea to use log pdf nor cdf '
                    f'in map {self.file_path}. '
                    f'Because its coordinate type is log-binned. '
                )

        if data['coordinate_name'] == 'pdf':
            warn(f'Convert {self.name} from (x, pdf) to (cdf, x).')
            x, cdf = self.pdf_to_cdf(data['coordinate_system'], data['map'])
            data['coordinate_name'] = 'cdf'
            data['coordinate_system'] = cdf
            data['map'] = x

        self.coordinate_type = data['coordinate_type']
        self.coordinate_name = data['coordinate_name']
        self.coordinate_system = jnp.asarray(data['coordinate_system'], dtype=float)
        self.map = jnp.asarray(data['map'], dtype=float)

        setattr(self, 'interpolator', interpolation.curve_interpolator)
        if self.coordinate_type == 'log_point':
            if jnp.any(self.coordinate_system <= 0):
                raise ValueError(
                    f'Find non-positive coordinate system in map {self.file_path}, '
                    f'which is specified as {self.coordinate_type}'
                )
            setattr(self, 'preprocess', self.log_pos)
        else:
            setattr(self, 'preprocess', self.linear_pos)
        setattr(self, 'apply', self.map_point)

    def map_point(self, pos):
        val = self.interpolator(
            self.preprocess(pos),
            self.preprocess(self.coordinate_system),
            self.preprocess(self.map),
        )
        return val

    def build_regbin(self, data):
        """Cache the map to jnp.array if bins_type is regbin"""
        if 'pdf' in data['coordinate_name'] or 'cdf' in data['coordinate_name']:
            if data['coordinate_type'] == 'log_regbin':
                raise ValueError(
                    f'It is not a good idea to use log pdf nor cdf '
                    f'in map {self.file_path}. '
                    f'Because its coordinate type is log-binned. '
                )

        self.coordinate_type = data['coordinate_type']
        self.coordinate_name = data['coordinate_name']
        self.coordinate_lowers = jnp.asarray(data['coordinate_lowers'], dtype=float)
        self.coordinate_uppers = jnp.asarray(data['coordinate_uppers'], dtype=float)
        self.map = jnp.asarray(data['map'], dtype=float)

        if len(self.coordinate_lowers) == 1:
            setattr(self, 'interpolator', interpolation.map_interpolator_regular_binning_1d)
        elif len(self.coordinate_lowers) == 2:
            setattr(self, 'interpolator', interpolation.map_interpolator_regular_binning_2d)
        elif len(self.coordinate_lowers) == 3:
            setattr(self, 'interpolator', interpolation.map_interpolator_regular_binning_3d)
        if self.coordinate_type == 'log_regbin':
            if jnp.any(self.coordinate_lowers <= 0) or jnp.any(self.coordinate_uppers <= 0):
                raise ValueError(
                    f'Find non-positive coordinate system in map {self.file_path}, '
                    f'which is specified as {self.coordinate_type}'
                )
            setattr(self, 'preprocess', self.log_pos)
        else:
            setattr(self, 'preprocess', self.linear_pos)
        setattr(self, 'apply', self.map_regbin)

    def map_regbin(self, pos):
        val = self.interpolator(
            self.preprocess(pos),
            self.preprocess(self.coordinate_lowers),
            self.preprocess(self.coordinate_uppers),
            self.map,
        )
        return val

    def linear_pos(self, pos):
        return pos

    def log_pos(self, pos):
        return jnp.log10(jnp.clip(pos, a_min=FLOAT_POS_MIN, a_max=FLOAT_POS_MAX))

    def pdf_to_cdf(self, x, pdf):
        """Convert pdf map to cdf map"""
        norm = integrate_midpoint(x, pdf)
        x, cdf = cum_integrate_midpoint(x, pdf)
        cdf /= norm
        return x, cdf


@export
class SigmaMap(Config):
    """
    Maps with uncertainty.
    Default value is a list whose order is:
    [median, lower, upper, (parameter)]
    Each map is assigned as attribute of SigmaMap.
    If the last element in the list is the required parameter.
    """

    def build(self, llh_name: str = None):
        """Read maps"""
        self.llh_name = llh_name
        if self.name in _cached_configs:
            _configs = _cached_configs[self.name]
        else:
            _configs = self.get_default()
            # Update values to sharing dictionary
            _cached_configs[self.name] = _configs

        if isinstance(_configs, dict):
            try:
                self._configs = _configs[llh_name]
            except KeyError:
                mesg = f'You specified {self.name} as a dictionary. '
                mesg += f'The key of it should be the name of one '
                mesg += f'of the likelihood, '
                mesg += f'but it is {llh_name}.'
                raise ValueError(mesg)
        else:
            self._configs = _configs

        self._configs_default = self.get_default()

        maps = {}
        sigmas = ['median', 'lower', 'upper']
        for i, sigma in enumerate(sigmas):
            maps[sigma] = Map(
                name=self.name + f'_{sigma}',
                default=self._configs_default[i])
            if maps[sigma].name not in _cached_configs.keys():
                _cached_configs[maps[sigma].name] = {}
            if isinstance(_cached_configs[maps[sigma].name], dict):
                # In case some plugins only use the median
                # and may already update the map name in `_cached_configs`
                _cached_configs[maps[sigma].name].update(
                    {self.llh_name: self._configs[i]})
            setattr(self, sigma, maps[sigma])

        self.median.build(llh_name=self.llh_name)
        self.lower.build(llh_name=self.llh_name)
        self.upper.build(llh_name=self.llh_name)

        if len(self._configs) > 4:
            raise ValueError(f'You give too much information in {self.name} configs.')

        # Find required parameter
        if len(self._configs) == 4:
            self.required_parameter = self._configs[-1]
            print(
                f'{self.llh_name} is using the parameter '
                f'{self.required_parameter} in {self.name} map.')
        else:
            self.required_parameter = self.name + '_sigma'

    def apply(self, pos, parameters):
        """Apply SigmaMap with sigma and position"""
        sigma = parameters[self.required_parameter]
        median = self.median.apply(pos)
        lower = self.lower.apply(pos)
        upper = self.upper.apply(pos)
        add_pos = (upper - median) * sigma
        add_neg = (median - lower) * sigma
        add = jnp.where(sigma > 0, add_pos, add_neg)
        return median + add
