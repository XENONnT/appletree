from typing import Optional, Union, Any

from immutabledict import immutabledict
from jax import numpy as jnp
from warnings import warn

import numpy as np

from appletree.share import _cached_configs
from appletree.utils import (
    exporter,
    load_json,
    get_file_path,
    integrate_midpoint,
    cum_integrate_midpoint,
)
from appletree import interpolation
from appletree.interpolation import FLOAT_POS_MIN, FLOAT_POS_MAX

export, __all__ = exporter()

OMITTED = "<OMITTED>"

__all__.extend(["OMITTED"])


@export
def takes_config(*configs):
    """Decorator for plugin classes, to specify which configs it takes.

    Args:
        configs: Config instances of configs this plugin takes.

    """

    def wrapped(plugin_class):
        """
        Args:
            plugin_class: plugin needs configuration
        """
        result = dict()
        for config in configs:
            if not isinstance(config, Config):
                raise RuntimeError("Specify config options by Config objects")
            config.taken_by = plugin_class.__name__
            result[config.name] = config

        if hasattr(plugin_class, "takes_config") and len(plugin_class.takes_config):
            # Already have some configs set, e.g. because of subclassing
            # where both child and parent have a takes_config decorator
            for config in result.values():
                if config.name in plugin_class.takes_config:
                    raise RuntimeError(f"Attempt to specify config {config.name} twice")
            plugin_class.takes_config = immutabledict({**plugin_class.takes_config, **result})
        else:
            plugin_class.takes_config = immutabledict(result)

        # Should set the configurations as the attributes of Plugin
        return plugin_class

    return wrapped


@export
class Config:
    """Configuration option taken by a appletree plugin."""

    llh_name: Optional[str] = None

    def __init__(
        self,
        name: str,
        type: Union[type, tuple, list, str] = OMITTED,
        default: Any = OMITTED,
        help: str = "",
    ):
        """Initialization.

        Args:
            name: name of the map.
            type: Excepted type of the option's value.
            default: Default value the option takes.
            help: description of the map.

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
        """Get default value of configuration."""
        if self.default is not OMITTED:
            return self.default

        raise ValueError(f"Missing option {self.name} required by {self.taken_by}")

    def build(self, llh_name: Optional[str] = None):
        """Build configuration, set attributes to Config instance."""
        raise NotImplementedError

    def required_parameter(self, llh_name=None):
        return None


@export
class Constant(Config):
    """Constant is a special config which takes only certain value."""

    value = None

    def build(self, llh_name: Optional[str] = None):
        """Set value of Constant."""
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
                mesg = (
                    f"You specified {self.name} as a dictionary. "
                    f"The key of it should be the name of one "
                    f"of the likelihood, but it is {llh_name}."
                )
                raise ValueError(mesg)
        else:
            self.value = value


@export
class Map(Config):
    """Map is a special config that takes input files.

    The method ``apply`` is dynamically assigned.
    When using points, the ``apply`` will be ``map_point``,
    while using regular binning, the ``apply`` will be ``map_regbin``.
    When using log-binning, we will first convert the positions to log space.

    """

    def build(self, llh_name: Optional[str] = None):
        """Cache the map to jnp.array."""

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
                mesg = (
                    f"You specified {self.name} as a dictionary. "
                    f"The key of it should be the name of one "
                    f"of the likelihood, but it is {llh_name}."
                )
                raise ValueError(mesg)
        else:
            self.file_path = file_path

        # try to find the path first
        _file_path = get_file_path(self.file_path)
        try:
            data = load_json(self.file_path)
        except Exception:
            raise ValueError(f"Cannot load {self.name} from {_file_path}!")

        coordinate_type = data["coordinate_type"]
        if coordinate_type == "point" or coordinate_type == "log_point":
            self.build_point(data)
        elif coordinate_type == "regbin" or coordinate_type == "log_regbin":
            self.build_regbin(data)
        else:
            raise ValueError("map_type must be either 'point' or 'regbin'!")

    def build_point(self, data):
        """Cache the map to jnp.array if bins_type is point."""
        if data["coordinate_name"] == "pdf" or data["coordinate_name"] == "cdf":
            if data["coordinate_type"] == "log_point":
                raise ValueError(
                    f"It is not a good idea to use log pdf nor cdf "
                    f"in map {self.file_path}. "
                    f"Because its coordinate type is log-binned. "
                )

        if data["coordinate_name"] == "pdf":
            warn(f"Convert {self.name} from (x, pdf) to (cdf, x).")
            x, cdf = self.pdf_to_cdf(data["coordinate_system"], data["map"])
            data["coordinate_name"] = "cdf"
            data["coordinate_system"] = cdf
            data["map"] = x

        self.coordinate_type = data["coordinate_type"]
        self.coordinate_name = data["coordinate_name"]
        self.coordinate_system = jnp.asarray(data["coordinate_system"], dtype=float)
        self.map = jnp.asarray(data["map"], dtype=float)

        setattr(self, "interpolator", interpolation.curve_interpolator)
        if self.coordinate_type == "log_point":
            if jnp.any(self.coordinate_system <= 0):
                raise ValueError(
                    f"Find non-positive coordinate system in map {self.file_path}, "
                    f"which is specified as {self.coordinate_type}"
                )
            setattr(self, "preprocess", self.log_pos)
        else:
            setattr(self, "preprocess", self.linear_pos)
        setattr(self, "apply", self.map_point)

    def map_point(self, pos):
        val = self.interpolator(
            self.preprocess(pos),
            self.preprocess(self.coordinate_system),
            self.preprocess(self.map),
        )
        return val

    def build_regbin(self, data):
        """Cache the map to jnp.array if bins_type is regbin."""
        if "pdf" in data["coordinate_name"] or "cdf" in data["coordinate_name"]:
            if data["coordinate_type"] == "log_regbin":
                raise ValueError(
                    f"It is not a good idea to use log pdf nor cdf "
                    f"in map {self.file_path}. "
                    f"Because its coordinate type is log-binned. "
                )

        self.coordinate_type = data["coordinate_type"]
        self.coordinate_name = data["coordinate_name"]
        self.coordinate_lowers = jnp.asarray(data["coordinate_lowers"], dtype=float)
        self.coordinate_uppers = jnp.asarray(data["coordinate_uppers"], dtype=float)
        self.map = jnp.asarray(data["map"], dtype=float)

        if len(self.coordinate_lowers) == 1:
            setattr(self, "interpolator", interpolation.map_interpolator_regular_binning_1d)
        elif len(self.coordinate_lowers) == 2:
            setattr(self, "interpolator", interpolation.map_interpolator_regular_binning_2d)
        elif len(self.coordinate_lowers) == 3:
            setattr(self, "interpolator", interpolation.map_interpolator_regular_binning_3d)
        if self.coordinate_type == "log_regbin":
            if jnp.any(self.coordinate_lowers <= 0) or jnp.any(self.coordinate_uppers <= 0):
                raise ValueError(
                    f"Find non-positive coordinate system in map {self.file_path}, "
                    f"which is specified as {self.coordinate_type}"
                )
            setattr(self, "preprocess", self.log_pos)
        else:
            setattr(self, "preprocess", self.linear_pos)
        setattr(self, "apply", self.map_regbin)

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
        """Convert pdf map to cdf map."""
        norm = integrate_midpoint(x, pdf)
        x, cdf = cum_integrate_midpoint(x, pdf)
        cdf /= norm
        return x, cdf


@export
class SigmaMap(Config):
    """Maps with uncertainty.

    The value of a SigmaMap can be:
     * a list with four elements,
        which are the file names of median, lower, upper maps and the name of the scaler.
     * a list with three elements,
        which are the file names of median, lower and upper maps. The name of the scaler
        is the default one f"{self.name}_sigma".
     * a string,
        which is the file name of the map for median, lower, upper.

    In the first and second case, the name of the scaler will appear
    in Component.needed_parameters.

    """

    def build(self, llh_name: Optional[str] = None):
        """Read maps."""
        self.llh_name = llh_name
        _configs = self.get_configs()

        _configs_default = self.get_default()

        if isinstance(_configs, list) and len(_configs) > 4:
            raise ValueError(f"You give too much information in {self.name}'s configs.")

        if isinstance(_configs_default, list) and len(_configs_default) > 4:
            raise ValueError(f"You give too much information in {self.name}'s default configs.")

        maps = dict()
        sigmas = ["median", "lower", "upper"]
        for i, sigma in enumerate(sigmas):
            # propagate _configs_default to Map instances
            if isinstance(_configs_default, list):
                default = _configs_default[i]
            else:
                if not isinstance(_configs_default, str):
                    raise ValueError(
                        f"If {self.name}'s default configuration is not a list, "
                        "then it should be a string."
                    )
                # If only one file is given, then use the same file for all sigmas
                default = _configs_default
            maps[sigma] = Map(name=self.name + f"_{sigma}", default=default)

            setattr(self, sigma, maps[sigma])

            if self.llh_name is None:
                # if llh_name is not specified, no need to update _cached_configs
                continue

            # In case some plugins only use the median
            # and may already update the map name in `_cached_configs`
            if maps[sigma].name not in _cached_configs.keys():
                _cached_configs[maps[sigma].name] = dict()
            if isinstance(_cached_configs[maps[sigma].name], dict):
                if isinstance(_configs, list):
                    value = _configs[i]
                else:
                    if not isinstance(_configs, str):
                        raise ValueError(
                            f"If {self.name}'s configuration is not a list, "
                            "then it should be a string."
                        )
                    # If only one file is given, then use the same file for all sigmas
                    value = _configs
                _value = _cached_configs[maps[sigma].name].get(self.llh_name, value)
                if _value != value:
                    raise ValueError(
                        f"You give different values for {self.name} in "
                        f"configs, find {_value} and {value}."
                    )
                _cached_configs[maps[sigma].name].update({self.llh_name: value})

        self.median.build(llh_name=self.llh_name)  # type: ignore
        self.lower.build(llh_name=self.llh_name)  # type: ignore
        self.upper.build(llh_name=self.llh_name)  # type: ignore

        required_parameter = self.required_parameter()
        if required_parameter is not None:
            print(
                f"{self.llh_name}'s map {self.name} is using "
                f"the parameter {required_parameter}."
            )
        else:
            print(f"{self.llh_name}'s map {self.name} is static and not using any parameter.")

    def get_configs(self, llh_name=None):
        """Get configs of SigmaMap."""
        # if llh_name is not specified, use the attribute of SigmaMap
        if llh_name is None and self.llh_name is not None:
            llh_name = self.llh_name

        if self.name in _cached_configs:
            _configs = _cached_configs[self.name]
        else:
            _configs = self.get_default()
            # Update values to sharing dictionary
            _cached_configs[self.name] = _configs

        if isinstance(_configs, dict):
            if llh_name is None:
                raise ValueError(
                    f"You specified {self.name} as a dictionary in _cached_configs. "
                    "The key of it should be the name of one of the likelihood, but it is None."
                )
            try:
                return _configs[llh_name]
            except KeyError:
                mesg = (
                    f"You specified {self.name} as a dictionary. "
                    f"The key of it should be the name of one "
                    f"of the likelihood, but it is {llh_name}."
                )
                raise ValueError(mesg)
        else:
            return _configs

    def required_parameter(self, llh_name=None):
        """Get required parameter of SigmaMap."""
        _configs = self.get_configs(llh_name=llh_name)
        # Find required parameter
        if isinstance(_configs, list):
            if len(_configs) == 4:
                return _configs[-1]
            else:
                return self.name + "_sigma"
        else:
            return None

    def apply(self, pos, parameters):
        """Apply SigmaMap with sigma and position."""
        if self.required_parameter() is None:
            sigma = 1.0
        else:
            sigma = parameters[self.required_parameter()]
        median = self.median.apply(pos)
        lower = self.lower.apply(pos)
        upper = self.upper.apply(pos)
        add_pos = (upper - median) * sigma
        add_neg = (median - lower) * sigma
        add = jnp.where(sigma > 0, add_pos, add_neg)
        return median + add


@export
class ConstantSet(Config):
    """ConstantSet is a special config which takes a set of values.

    We will not specify any hard-coded distribution or function here. User should be careful with
    the actual function implemented. Fortunately, we only use these values as keyword arguments, so
    mismatch will be catched when running.

    """

    def build(self, llh_name: Optional[str] = None):
        """Set value of Constant."""
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
                mesg = (
                    f"You specified {self.name} as a dictionary. "
                    f"The key of it should be the name of one "
                    f"of the likelihood, but it is {llh_name}."
                )
                raise ValueError(mesg)
        else:
            self.value = value

        self._sanity_check()
        self.set_volume = len(self.value[1][0])
        self.value = {k: jnp.array(v) for k, v in zip(*self.value)}

    def _sanity_check(self):
        """Check if parameter set lengths are same."""
        mesg = "The given values should follow [names, values] format."
        assert len(self.value) == 2, mesg
        mesg = "Parameters and their names should have same length"
        assert len(self.value[0]) == len(self.value[1]), mesg
        volumes = [len(v) for v in self.value[1]]
        mesg = "Parameter set lengths should be the same"
        assert np.all(np.isclose(volumes, volumes[0])), mesg
