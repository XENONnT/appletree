from warnings import warn
from functools import partial
from typing import Tuple, List, Dict, Optional, Union, Set

import numpy as np
import pandas as pd
from jax import numpy as jnp

from appletree import utils
from appletree.config import OMITTED
from appletree.plugin import Plugin
from appletree.share import _cached_configs, _cached_functions, set_global_config
from appletree.utils import exporter, load_data
from appletree.hist import make_hist_mesh_grid, make_hist_irreg_bin_1d, make_hist_irreg_bin_2d

export, __all__ = exporter()


@export
class Component:
    """Base class of component."""

    # Do not initialize this class because it is base
    __is_base = True

    rate_name: str = ""
    norm_type: str = ""
    # add_eps_to_hist==True was introduced as only a workaround
    # for likelihood blowup problem when using meshgrid binning
    add_eps_to_hist: bool = True
    force_no_eff: bool = False

    def __init__(self, name: Optional[str] = None, llh_name: Optional[str] = None, **kwargs):
        """Initialization.

        Args:
            bins: bins to generate the histogram.
                * For irreg bins_type, bins must be bin edges of the two dimensions.
                * For meshgrid bins_type, bins are sent to jnp.histogramdd.
            bins_type: binning scheme, can be either irreg or meshgrid.

        """
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        if llh_name is None:
            self.llh_name = self.__class__.__name__ + "_llh"
        else:
            self.llh_name = llh_name
        self.needed_parameters: Set[str] = set()

        if "bins" in kwargs.keys() and "bins_type" in kwargs.keys():
            self.set_binning(**kwargs)

            if self.bins_type != "meshgrid" and self.add_eps_to_hist:
                warn(
                    "It is empirically dangerous to have add_eps_to_hist==True, "
                    "when your bins_type is not meshgrid! It may lead to very bad fit with "
                    "lots of eff==0."
                )

    def set_binning(self, **kwargs):
        """Set binning of component."""
        if "bins" not in kwargs.keys() or "bins_type" not in kwargs.keys():
            raise ValueError("bins and bins_type must be set!")
        self.bins = kwargs.get("bins")
        self.bins_type = kwargs.get("bins_type")
        if self.bins_type not in ["irreg", "meshgrid", None]:
            raise ValueError(f"Unsupported bins_type {self.bins_type}!")

        if self.bins_type == "meshgrid":
            warning = "The usage of meshgrid binning is highly discouraged."
            warn(warning)

    def _clip(self, result: list):
        """Clip simulated result."""
        mask = np.ones(len(result[-1]), dtype=bool)
        for i in range(len(result) - 1):
            mask &= result[i] > np.array(self.bins[i]).min()
            mask &= result[i] < np.array(self.bins[i]).max()
        for i in range(len(result)):
            result[i] = result[i][mask]
        return result

    @property
    def _use_mcinput(self):
        return "Bootstrap" in self._plugin_class_registry["energy"].__name__

    def simulate_hist(self, *args, **kwargs):
        """Hook for simulation with histogram output."""
        raise NotImplementedError

    def multiple_simulations(self, key, batch_size, parameters, times, apply_eff=False):
        """Simulate many times and move results to CPU because the memory limit of GPU."""
        results_pile = []
        assert times > 0, "times of multiple simulations must be greater than 0!"
        for _ in range(times):
            key, results = self.simulate(key, batch_size, parameters)
            if apply_eff:
                if self.force_no_eff:
                    raise RuntimeError(
                        "You are forcing to apply efficiency! "
                        "But component was set to not returning efficiency when "
                        f"running {self.name}.deduce!"
                    )
                mask = np.array(results[-1]) > 0
                for i in range(len(results)):
                    results[i] = np.array(results[i])[mask]
            results_pile.append(results)
        results_pile = [
            np.hstack([results_pile[j][i] for j in range(times)]) for i in range(len(results))
        ]
        return key, results_pile

    def multiple_simulations_compile(self, key, batch_size, parameters, times, apply_eff=False):
        """Simulate many times after new compilation and move results to CPU because the memory
        limit of GPU."""
        results_pile = []
        for _ in range(times):
            if _cached_configs["g4"] and self._use_mcinput:
                if isinstance(_cached_configs["g4"], dict):
                    g4_file_name = _cached_configs["g4"][self.llh_name][0]
                    _cached_configs["g4"][self.llh_name] = [
                        g4_file_name,
                        batch_size,
                        key.sum().item(),
                    ]
                else:
                    g4_file_name = _cached_configs["g4"][0]
                    _cached_configs["g4"] = [g4_file_name, batch_size, key.sum().item()]
            self.compile()
            key, results = self.multiple_simulations(key, batch_size, parameters, 1, apply_eff)
            results_pile.append(results)
        results_pile = [
            np.hstack([results_pile[j][i] for j in range(times)]) for i in range(len(results))
        ]
        return key, results_pile

    def implement_binning(self, mc, eff):
        """Apply binning to MC data.

        Args:
            mc: data from simulation.
            eff: efficiency of each event, as the weight when making a histogram.

        """
        if self.bins_type == "irreg":
            if len(self.bins) == 1:
                hist = make_hist_irreg_bin_1d(mc[:, 0], self.bins[0], weights=eff)
            elif len(self.bins) == 2:
                hist = make_hist_irreg_bin_2d(mc, *self.bins, weights=eff)
            else:
                raise ValueError(f"Currently only support 1D and 2D, but got {len(self.bins)}D!")
        elif self.bins_type == "meshgrid":
            hist = make_hist_mesh_grid(mc, bins=self.bins, weights=eff)
        else:
            raise ValueError(f"Unsupported bins_type {self.bins_type}!")
        if self.add_eps_to_hist:
            # as an uncertainty to prevent blowing up
            hist = jnp.clip(hist, 1.0, jnp.inf)
        return hist

    def get_normalization(self, hist, parameters, batch_size=None):
        """Return the normalization factor of the histogram."""
        if self.norm_type == "on_pdf":
            normalization_factor = 1 / jnp.sum(hist) * parameters[self.rate_name]
        elif self.norm_type == "on_sim":
            if self._use_mcinput:
                bootstrap_name = self._plugin_class_registry["energy"].__name__
                bootstrap_name = bootstrap_name + "_" + self.name
                n_events_selected = _cached_functions[self.llh_name][
                    bootstrap_name
                ].g4.n_events_selected
                normalization_factor = 1 / n_events_selected * parameters[self.rate_name]
            else:
                normalization_factor = 1 / batch_size * parameters[self.rate_name]
        else:
            raise ValueError(f"Unsupported norm_type {self.norm_type}!")
        return normalization_factor

    def deduce(self, *args, **kwargs):
        """Hook for workflow deduction."""
        raise NotImplementedError

    def compile(self):
        """Hook for compiling simulation code."""
        pass


@export
class ComponentSim(Component):
    """Component that needs MC simulations."""

    # Do not initialize this class because it is base
    __is_base = True

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)
        self._plugin_class_registry = dict()

    def register(self, plugin_class):
        """Register a plugin to the component."""
        if isinstance(plugin_class, (tuple, list)):
            # Shortcut for multiple registration
            for x in plugin_class:
                self.register(x)
            return

        # Ensure plugin_class.provides is a tuple
        if isinstance(plugin_class.provides, str):
            plugin_class.provides = tuple([plugin_class.provides])

        for p in plugin_class.provides:
            self._plugin_class_registry[p] = plugin_class

        already_seen = []
        for plugin in self._plugin_class_registry.values():
            if plugin in already_seen:
                continue
            already_seen.append(plugin)

            for config_name, items in plugin.takes_config.items():
                # Looping over the configs of the new plugin and check if
                # they can be found in the already registered plugins:
                for new_config, new_items in plugin_class.takes_config.items():
                    if new_config != config_name:
                        continue
                    if items.default == new_items.default:
                        continue
                    else:
                        mes = (
                            f"Two plugins have a different file name "
                            f"for the same config. The config "
                            f"'{new_config}' in '{plugin.__name__}' takes "
                            f"the file name as '{new_items.default}'  while in "
                            f"'{plugin_class.__name__}' the file name "
                            f"is set to '{items.default}'. Please change "
                            f"one of the file names."
                        )
                        raise ValueError(mes)

    def register_all(self, module):
        """Register all plugins defined in module.

        Can pass a list/tuple of modules to register all in each.

        """
        if isinstance(module, (tuple, list)):
            # Shortcut for multiple registration
            for x in module:
                self.register_all(x)
            return

        for x in dir(module):
            x = getattr(module, x)
            if not isinstance(x, type(type)):
                continue
            if issubclass(x, Plugin):
                self.register(x)

    def dependencies_deduce(
        self,
        data_names: Union[List[str], Tuple[str]] = ["cs1", "cs2", "eff"],
        dependencies: Optional[List[Dict]] = None,
        nodep_data_name: str = "batch_size",
    ) -> list:
        """Deduce dependencies.

        Args:
            data_names: data names that simulation will output.
            dependencies: dependency tree.
            nodep_data_name: data_name without dependency will not be deduced.

        """
        if dependencies is None:
            dependencies = []

        for data_name in data_names:
            # usually `batch_size` have no dependency
            if data_name == nodep_data_name:
                continue
            try:
                dependencies.append(
                    {
                        "plugin": self._plugin_class_registry[data_name],
                        "provides": data_name,
                        "depends_on": self._plugin_class_registry[data_name].depends_on,
                    }
                )
            except KeyError:
                raise ValueError(f"Can not find dependency for {data_name}")

        for data_name in data_names:
            # `batch_size` has no dependency
            if data_name == nodep_data_name:
                continue
            dependencies = self.dependencies_deduce(
                data_names=self._plugin_class_registry[data_name].depends_on,
                dependencies=dependencies,
                nodep_data_name=nodep_data_name,
            )

        return dependencies

    def dependencies_simplify(self, dependencies):
        """Simplify the dependencies."""
        already_seen = []
        self.worksheet = []
        # Reinitialize needed_parameters
        # because sometimes user will deduce(& compile) after changing configs
        self.needed_parameters: Set[str] = set()
        # Add rate_name to needed_parameters only when it's not empty
        if self.rate_name != "":
            self.needed_parameters.add(self.rate_name)
        for _plugin in dependencies[::-1]:
            plugin = _plugin["plugin"]
            if plugin.__name__ in already_seen:
                continue
            self.worksheet.append([plugin.__name__, plugin.provides, plugin.depends_on])
            already_seen.append(plugin.__name__)
            self.needed_parameters |= set(plugin.parameters)
            # Add needed_parameters from config
            for config in plugin.takes_config.values():
                required_parameter = config.required_parameter(self.llh_name)
                if required_parameter is not None:
                    self.needed_parameters |= {required_parameter}

    def flush_source_code(
        self,
        data_names: Union[List[str], Tuple[str]] = ["cs1", "cs2", "eff"],
        func_name: str = "simulate",
        nodep_data_name: str = "batch_size",
    ):
        """Infer the simulation code from the dependency tree."""
        self.func_name = func_name

        if not isinstance(data_names, (list, str)):
            raise RuntimeError(f"data_names must be list or str, but given {type(data_names)}")
        if isinstance(data_names, str):
            data_names = [data_names]

        code = ""
        indent = " " * 4

        code += "from functools import partial\n"
        code += "from jax import jit\n"

        # import needed plugins
        for work in self.worksheet:
            plugin = work[0]
            code += f"from appletree.plugins import {plugin}\n"

        # initialize new instances
        for work in self.worksheet:
            plugin = work[0]
            instance = plugin + "_" + self.name
            code += f"{instance} = {plugin}('{self.llh_name}')\n"

        # define functions
        code += "\n"
        if nodep_data_name == "batch_size":
            code += "@partial(jit, static_argnums=(1, ))\n"
        else:
            code += "@jit\n"
        code += f"def {func_name}(key, {nodep_data_name}, parameters):\n"

        for work in self.worksheet:
            provides = "key, " + ", ".join(work[1])
            depends_on = ", ".join(work[2])
            instance = work[0] + "_" + self.name
            code += f"{indent}{provides} = {instance}(key, parameters, {depends_on})\n"
        output = "key, " + "[" + ", ".join(data_names) + "]"
        code += f"{indent}return {output}\n"

        self.code = code

        if func_name in _cached_functions[self.llh_name].keys():
            warning = f"Function name {func_name} is already cached. "
            warning += "Running compile() will overwrite it."
            warn(warning)

    @property
    def code(self):
        """Code of simulation function."""
        return self._code

    @code.setter
    def code(self, code):
        self._code = code
        _cached_functions[self.llh_name] = dict()
        self._compile = partial(exec, self.code, _cached_functions[self.llh_name])

    def deduce(
        self,
        data_names: Union[List[str], Tuple[str]] = ["cs1", "cs2"],
        func_name: str = "simulate",
        nodep_data_name: str = "batch_size",
        force_no_eff: bool = False,
    ):
        """Deduce workflow and code.

        Args:
            data_names: data names that simulation will output.
            func_name: name of the simulation function, used to cache it.
            nodep_data_name: data_name without dependency will not be deduced.
            force_no_eff: force to ignore the efficiency, used in yield prediction.

        """
        if not isinstance(data_names, (list, tuple)):
            raise ValueError(f"Unsupported data_names type {type(data_names)}!")
        # make sure that 'eff' is the last data_name
        data_names = list(data_names)
        if "eff" in data_names:
            data_names.remove("eff")
        if not force_no_eff:
            data_names += ["eff"]
        else:
            # track status of component
            self.force_no_eff = True

        dependencies = self.dependencies_deduce(data_names, nodep_data_name=nodep_data_name)
        self.dependencies_simplify(dependencies)
        self.flush_source_code(data_names, func_name, nodep_data_name)

    def compile(self):
        """Build simulation function and cache it to share._cached_functions."""
        self._compile()
        self.simulate = _cached_functions[self.llh_name][self.func_name]

    def simulate_hist(self, key, batch_size, parameters):
        """Simulate and return histogram.

        Args:
            key: key used for pseudorandom generator.
            batch_size: number of events to be simulated.
            parameters: a dictionary that contains all parameters needed in simulation.

        """
        key, result = self.simulate(key, batch_size, parameters)
        if self.force_no_eff:
            mc = jnp.asarray(result).T
            eff = jnp.ones(mc.shape[0])
        else:
            mc = jnp.asarray(result[:-1]).T
            eff = result[-1]  # we guarantee that the last output is efficiency in self.deduce
        assert mc.shape[1] == len(
            self.bins
        ), "Length of bins must be the same as length of bins_on!"

        hist = self.implement_binning(mc, eff)
        normalization_factor = self.get_normalization(hist, parameters, batch_size)
        hist *= normalization_factor

        return key, hist

    def simulate_weighted_data(self, key, batch_size, parameters):
        """Simulate and return histogram."""
        key, result = self.simulate(key, batch_size, parameters)
        # Move data to CPU
        result = [np.array(r) for r in result]
        # Clip data points out of ROI
        result = self._clip(result)
        mc = result[:-1]
        assert len(mc) == len(self.bins), "Length of bins must be the same as length of bins_on!"
        mc = jnp.asarray(mc).T
        eff = jnp.asarray(
            result[-1]
        )  # we guarantee that the last output is efficiency in self.deduce

        hist = self.implement_binning(mc, eff)
        normalization_factor = self.get_normalization(hist, parameters, batch_size)
        result[-1] *= normalization_factor

        return key, result

    def save_code(self, file_path):
        """Save the code to file."""
        with open(file_path, "w") as f:
            f.write(self.code)

    def lineage(self, data_name: str = "cs2"):
        """Return lineage of plugins."""
        assert isinstance(data_name, str)
        pass

    def set_config(self, configs):
        """Set new global configuration options.

        Args:
            configs: dict, configuration file name or dictionary

        """
        set_global_config(configs)
        warn(
            "New config is set, please run deduce() "
            "and compile() again to update the simulation code."
        )

    def show_config(self, data_names: Union[List[str], Tuple[str]] = ["cs1", "cs2", "eff"]):
        """Return configuration options that affect data_names.

        Args:
            data_names: Data type name

        """
        dependencies = self.dependencies_deduce(
            data_names,
            nodep_data_name="batch_size",
        )
        r = []
        seen = []

        for dep in dependencies:
            p = dep["plugin"]
            # Track plugins we already saw, so options from
            # multi-output plugins don't come up several times
            if p in seen:
                continue
            seen.append(p)

            for config in p.takes_config.values():
                try:
                    default = config.get_default()
                except ValueError:
                    default = OMITTED
                current = _cached_configs.get(config.name, None)
                if isinstance(current, dict):
                    current = current[self.llh_name]
                r.append(
                    dict(
                        option=config.name,
                        default=default,
                        current=current,
                        applies_to=p.provides,
                        help=config.help,
                    )
                )
        if len(r):
            df = pd.DataFrame(r, columns=r[0].keys())
        else:
            df = pd.DataFrame([])

        # Then you can print the dataframe like:
        # straxen.dataframe_to_wiki(df, title=f'{data_names}', float_digits=1)
        return df

    def new_component(self, llh_name: Optional[str] = None, pass_binning: bool = True):
        """Generate new component with same binning, usually used on predicting yields."""
        if pass_binning:
            if hasattr(self, "bins") and hasattr(self, "bins_type"):
                component = self.__class__(
                    name=self.name + "_copy",
                    llh_name=llh_name,
                    bins=self.bins,
                    bins_type=self.bins_type,
                )
            else:
                raise ValueError("Should provide bins and bins_type if you want to pass binning!")
        else:
            component = self.__class__(
                name=self.name + "_copy",
                llh_name=llh_name,
            )
        return component


@export
class ComponentFixed(Component):
    """Component whose shape is fixed."""

    # Do not initialize this class because it is base
    __is_base = True

    def __init__(self, *args, **kwargs):
        """Initialization."""
        if not kwargs.get("file_name", None):
            raise ValueError("Should provide file_name for ComponentFixed!")
        else:
            self._file_name = kwargs.get("file_name", None)
        super().__init__(*args, **kwargs)

    def deduce(self, data_names: Union[List[str], Tuple[str]] = ["cs1", "cs2"]):
        """Deduce the needed parameters and make the fixed histogram."""
        self.data = load_data(self._file_name)[list(data_names)].to_numpy()
        self.eff = jnp.ones(len(self.data))
        self.hist = self.implement_binning(self.data, self.eff)
        self.needed_parameters.add(self.rate_name)

    def simulate(self):
        """Fixed component does not need to simulate."""
        raise NotImplementedError

    def simulate_hist(self, parameters, *args, **kwargs):
        """Return the fixed histogram."""
        normalization_factor = self.get_normalization(self.hist, parameters, len(self.data))
        return self.hist * normalization_factor

    def simulate_weighted_data(self, parameters, *args, **kwargs):
        """Simulate and return histogram."""
        result = [r for r in self.data.T]
        result.append(np.array(self.eff))
        # Clip all simulated data points
        result = self._clip(result)
        normalization_factor = self.get_normalization(self.hist, parameters, len(self.data))
        result[-1] *= normalization_factor

        return result


@export
def add_component_extensions(module1, module2, force=False):
    """Add components of module2 to module1."""
    utils.add_extensions(module1, module2, Component, force=force)


@export
def _add_component_extension(module, component, force=False):
    """Add component to module."""
    utils._add_extension(module, component, Component, force=force)
