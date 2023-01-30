from warnings import warn
from functools import partial

import numpy as np
import pandas as pd
from jax import numpy as jnp
import pandas as pd

import appletree
from appletree import utils
from appletree.plugin import Plugin
from appletree.share import _cached_configs, _cached_functions, set_global_config
from appletree.utils import exporter, load_data
from appletree.hist import make_hist_mesh_grid, make_hist_irreg_bin_2d

export, __all__ = exporter()


@export
class Component:
    """Base class of component"""

    # Do not initialize this class because it is base
    __is_base = True

    rate_name: str = ''
    norm_type: str = ''
    add_eps_to_hist: bool = True

    def __init__(self,
                 name: str = None,
                 llh_name: str = None,
                 bins: list = [],
                 bins_type: str = '',
                 **kwargs):
        """Initialization.

        :param bins: bins to generate the histogram.

          * For irreg bins_type, bins must be bin edges of the two dimensions.
          * For meshgrid bins_type, bins are sent to jnp.histogramdd.
        :param bins_type: binning scheme, can be either irreg or meshgrid.
        """
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        if llh_name is None:
            self.llh_name = self.__class__.__name__ + '_llh'
        else:
            self.llh_name = llh_name
        self.bins = bins
        self.bins_type = bins_type
        self.needed_parameters = set()

        if self.bins_type == 'meshgrid':
            warning = 'The usage of meshgrid binning is highly discouraged.'
            warn(warning)

    def _clip(self, result: list):
        """Clip simulated result"""
        mask = np.ones(len(result[-1]), dtype=bool)
        for i in range(len(result) - 1):
            mask &= result[i] > np.array(self.bins[i]).min()
            mask &= result[i] < np.array(self.bins[i]).max()
        for i in range(len(result)):
            result[i] = result[i][mask]
        return result

    @property
    def _use_mcinput(self):
        return 'Bootstrap' in self._plugin_class_registry['energy'].__name__

    def simulate_hist(self, *args, **kwargs):
        """Hook for simulation with histogram output."""
        raise NotImplementedError

    def multiple_simulations(self, key, batch_size, parameters, times):
        """Simulate many times and
        move results to CPU because the memory limit of GPU
        """
        results_pile = []
        for _ in range(times):
            key, results = self.simulate(key, batch_size, parameters)
            results_pile.append(np.array(results))
        return key, np.hstack(results_pile)

    def multiple_simulations_compile(self, key, batch_size, parameters, times):
        """Simulate many times after new compilation and
        move results to CPU because the memory limit of GPU
        """
        results_pile = []
        for _ in range(times):
            if _cached_configs['g4'] and self._use_mcinput:
                if isinstance(_cached_configs['g4'], dict):
                    g4_file_name = _cached_configs['g4'][self.llh_name][0]
                    _cached_configs['g4'][self.llh_name] = [
                        g4_file_name, batch_size, key.sum().item()]
                else:
                    g4_file_name = _cached_configs['g4'][0]
                    _cached_configs['g4'] = [g4_file_name, batch_size, key.sum().item()]
            self.compile()
            key, results = self.multiple_simulations(key, batch_size, parameters, 1)
            results_pile.append(results)
        return key, np.hstack(results_pile)

    def implement_binning(self, mc, eff):
        """Apply binning to MC data.

        :param mc: data from simulation.
        :param eff: efficiency of each event, as the weight when making a histogram.
        """
        if self.bins_type == 'irreg':
            hist = make_hist_irreg_bin_2d(mc, *self.bins, weights=eff)
        elif self.bins_type == 'meshgrid':
            hist = make_hist_mesh_grid(mc, bins=self.bins, weights=eff)
        else:
            raise ValueError(f'Unsupported bins_type {self.bins_type}!')
        if self.add_eps_to_hist:
            # as an uncertainty to prevent blowing up
            hist = jnp.clip(hist, 1., jnp.inf)
        return hist

    def get_normalization(self, hist, parameters, batch_size=None):
        """Return the normalization factor of the histogram."""
        if self.norm_type == 'on_pdf':
            normalization_factor = 1 / jnp.sum(hist) * parameters[self.rate_name]
        elif self.norm_type == 'on_sim':
            if self._use_mcinput:
                bootstrap_name = self._plugin_class_registry['energy'].__name__
                bootstrap_name = bootstrap_name + '_' + self.name
                n_events_selected = _cached_functions[self.llh_name][bootstrap_name].g4.n_events_selected
                normalization_factor = 1 / n_events_selected * parameters[self.rate_name]
            else:
                normalization_factor = 1 / batch_size * parameters[self.rate_name]
        else:
            raise ValueError(f'Unsupported norm_type {self.norm_type}!')
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

    code: str = None
    old_code: str = None

    def __init__(self,
                 *args, **kwargs):
        """Initialization"""
        super().__init__(*args, **kwargs)
        self._plugin_class_registry = {}

    def register(self, plugin_class):
        """Register a plugin to the component."""
        if isinstance(plugin_class, (tuple, list)):
            # Shortcut for multiple registration
            for x in plugin_class:
                self.register(x)
            return

        if not hasattr(plugin_class, 'provides'):
            # No output name specified: construct one from the class name
            snake_name = appletree.camel_to_snake(plugin_class.__name__)
            plugin_class.provides = (snake_name,)

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
                        mes = f'Two plugins have a different file name'
                        mes += f' for the same config. The config'
                        mes += f' "{new_config}" in "{plugin.__name__}" takes'
                        mes += f' the file name as "{new_items.default}"  while in'
                        mes += f' "{plugin_class.__name__}" the file name'
                        mes += f' is set to "{items.default}". Please change'
                        mes += f' one of the file names.'
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

    def dependencies_deduce(self,
                            data_names: list = ('cs1', 'cs2', 'eff'),
                            dependencies: list = None,
                            nodep_data_name: str = 'batch_size') -> list:
        """Deduce dependencies.

        :param data_names: data names that simulation will output.
        :param dependencies: dependency tree.
        :param nodep_data_name: data_name without dependency will not be deduced
        """
        if dependencies is None:
            dependencies = []

        for data_name in data_names:
            # usually `batch_size` have no dependency
            if data_name == nodep_data_name:
                continue
            try:
                dependencies.append({
                    'plugin': self._plugin_class_registry[data_name],
                    'provides': data_name,
                    'depends_on': self._plugin_class_registry[data_name].depends_on,
                })
            except KeyError:
                raise ValueError(f'Can not find dependency for {data_name}')

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
        self.needed_parameters.add(self.rate_name)
        for plugin in dependencies[::-1]:
            plugin = plugin['plugin']
            if plugin.__name__ in already_seen:
                continue
            self.worksheet.append([plugin.__name__, plugin.provides, plugin.depends_on])
            already_seen.append(plugin.__name__)
            self.needed_parameters |= set(plugin.parameters)

    def flush_source_code(self,
                          data_names: list = ['cs1', 'cs2', 'eff'],
                          func_name: str = 'simulate',
                          nodep_data_name: str = 'batch_size'):
        """Infer the simulation code from the dependency tree."""
        self.func_name = func_name

        if not isinstance(data_names, (list, str)):
            raise RuntimeError(f'data_names must be list or str, but given {type(data_names)}')
        if isinstance(data_names, str):
            data_names = [data_names]

        code = ''
        indent = ' ' * 4

        code += 'from functools import partial\n'
        code += 'from jax import jit\n'

        # import needed plugins
        for work in self.worksheet:
            plugin = work[0]
            code += f'from appletree.plugins import {plugin}\n'

        # initialize new instances
        for work in self.worksheet:
            plugin = work[0]
            instance = plugin + '_' + self.name
            code += f"{instance} = {plugin}('{self.llh_name}')\n"

        # define functions
        code += '\n'
        if nodep_data_name == 'batch_size':
            code += '@partial(jit, static_argnums=(1, ))\n'
        else:
            code += '@jit\n'
        code += f'def {func_name}(key, {nodep_data_name}, parameters):\n'

        for work in self.worksheet:
            provides = 'key, ' + ', '.join(work[1])
            depends_on = ', '.join(work[2])
            instance = work[0] + '_' + self.name
            code += f'{indent}{provides} = {instance}(key, parameters, {depends_on})\n'
        output = 'key, ' + '[' + ', '.join(data_names) + ']'
        code += f'{indent}return {output}\n'

        self.code = code

        if func_name in _cached_functions[self.llh_name].keys():
            warning = f'Function name {func_name} is already cached. '
            warning += 'Running compile() will overwrite it.'
            warn(warning)

    @property
    def code(self):
        """Code of simulation function."""
        return self._code

    @code.setter
    def code(self, code):
        self._code = code
        if self.llh_name not in _cached_functions.keys():
            _cached_functions[self.llh_name] = {}
        self._compile = partial(exec, self.code, _cached_functions[self.llh_name])

    def deduce(self,
               data_names: list = ('cs1', 'cs2'),
               func_name: str = 'simulate',
               nodep_data_name: str = 'batch_size',
               force_no_eff: bool = False):
        """Deduce workflow and code.

        :param data_names: data names that simulation will output.
        :param func_name: name of the simulation function, used to cache it.
        :param nodep_data_name: data_name without dependency will not be deduced
        :param force_no_eff: force to ignore the efficiency, used in yield prediction
        """
        if not isinstance(data_names, (list, tuple)):
            raise ValueError(f'Unsupported data_names type {type(data_names)}!')
        # make sure that 'eff' is the last data_name
        if 'eff' in data_names:
            data_names = list(data_names)
            data_names.remove('eff')
        if not force_no_eff:
            data_names = list(data_names) + ['eff']

        dependencies = self.dependencies_deduce(data_names, nodep_data_name=nodep_data_name)
        self.dependencies_simplify(dependencies)
        self.flush_source_code(data_names, func_name, nodep_data_name)

    def compile(self):
        """Build simulation function and cache it to share._cached_functions."""
        self._compile()
        self.simulate = _cached_functions[self.llh_name][self.func_name]

    def simulate_hist(self,
                      key,
                      batch_size,
                      parameters):
        """Simulate and return histogram.

        :param key: key used for pseudorandom generator.
        :param batch_size: number of events to be simulated.
        :param parameters: a dictionary that contains all parameters needed in simulation.
        """
        key, result = self.simulate(key, batch_size, parameters)
        mc = result[:-1]
        assert len(mc) == len(self.bins), "Length of bins must be the same as length of bins_on!"
        mc = jnp.asarray(mc).T
        eff = result[-1]  # we guarantee that the last output is efficiency in self.deduce

        hist = self.implement_binning(mc, eff)
        normalization_factor = self.get_normalization(hist, parameters, batch_size)
        hist *= normalization_factor

        return key, hist

    def simulate_weighed_data(self,
                              key,
                              batch_size,
                              parameters):
        """Simulate and return histogram."""
        key, result = self.simulate(key, batch_size, parameters)
        # Move data to CPU
        result = [np.array(r) for r in result]
        # Clip data points out of ROI
        result = self._clip(result)
        mc = result[:-1]
        assert len(mc) == len(self.bins), "Length of bins must be the same as length of bins_on!"
        mc = jnp.asarray(mc).T
        eff = jnp.asarray(result[-1])  # we guarantee that the last output is efficiency in self.deduce

        hist = self.implement_binning(mc, eff)
        normalization_factor = self.get_normalization(hist, parameters, batch_size)
        result[-1] *= normalization_factor

        return key, result

    def save_code(self, file_path):
        """Save the code to file."""
        with open(file_path, 'w') as f:
            f.write(self.code)

    def lineage(self, data_name: str = 'cs2'):
        """Return lineage of plugins."""
        assert isinstance(data_name, str)
        pass

    def set_config(self, configs):
        """Set new global configuration options

        :param configs: dict, configuration file name or dictionary
        """
        set_global_config(configs)

    def show_config(self, data_names: list = ('cs1', 'cs2', 'eff')):
        """
        Return configuration options that affect data_names.

        :param data_names: Data type name
        """
        dependencies = self.dependencies_deduce(
            data_names,
            nodep_data_name='batch_size',
        )
        r = []
        seen = []

        for dep in dependencies:
            p = dep['plugin']
            # Track plugins we already saw, so options from
            # multi-output plugins don't come up several times
            if p in seen:
                continue
            seen.append(p)

            for config in p.takes_config.values():
                try:
                    default = config.get_default()
                except:
                    default = appletree.OMITTED
                current = _cached_configs.get(config.name, None)
                if isinstance(current, dict):
                    current = current[self.llh_name]
                r.append(dict(
                    option=config.name,
                    default=default,
                    current=current,
                    applies_to=p.provides,
                    help=config.help))
        if len(r):
            df = pd.DataFrame(r, columns=r[0].keys())
        else:
            df = pd.DataFrame([])

        # Then you can print the dataframe like:
        # straxen.dataframe_to_wiki(df, title=f'{data_names}', float_digits=1)
        return df

    def new_component(self):
        """
        Generate new component with same binning,
        usually used on predicting yields
        """
        component = self.__class__(
            name=self.name + '_copy',
            bins=self.bins,
            bins_type=self.bins_type,
        )
        return component


@export
class ComponentFixed(Component):
    """Component whose shape is fixed."""

    # Do not initialize this class because it is base
    __is_base = True

    def __init__(self,
                 *args, **kwargs):
        """Initialization"""
        if not kwargs.get('file_name', None):
            raise ValueError('Should provide file_name for ComponentFixed!')
        else:
            self._file_name = kwargs.get('file_name', None)
        super().__init__(*args, **kwargs)

    def deduce(self,
               data_names: list = ('cs1', 'cs2')):
        """Deduce the needed parameters and make the fixed histogram."""
        self.data = load_data(self._file_name)[list(data_names)].to_numpy()
        self.eff = jnp.ones(len(self.data))
        self.hist = self.implement_binning(self.data, self.eff)
        self.needed_parameters.add(self.rate_name)

    def simulate(self):
        """Fixed component does not need to simulate."""
        raise NotImplementedError

    def simulate_hist(self,
                      parameters,
                      *args, **kwargs):
        """Return the fixed histogram."""
        normalization_factor = self.get_normalization(self.hist, parameters, len(self.data))
        return self.hist * normalization_factor

    def simulate_weighed_data(self,
                              parameters,
                              *args, **kwargs):
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
    """Add components of module2 to module1"""
    utils.add_extensions(module1, module2, Component, force=force)


@export
def _add_component_extension(module, component, force=False):
    """Add component to module"""
    utils._add_extension(module, component, Component, force=force)
