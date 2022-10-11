from warnings import warn
from functools import partial
import pandas as pd
from jax import numpy as jnp
import pandas as pd

import appletree
from appletree.plugin import Plugin
from appletree.share import _cached_configs, _cached_functions
from appletree.utils import exporter, load_data
from appletree.hist import make_hist_mesh_grid, make_hist_irreg_bin_2d

export, __all__ = exporter()


@export
class Component:
    """Component base class"""

    rate_name: str = ''
    norm_type: str = ''
    tag: str = '_'  # for instance name of the plugins

    def __init__(self,
                 bins: list = [],
                 bins_type: str = '',
                 **kwargs):
        """Initialization.
        :param bins: bins to generate the histogram.
        For irreg bins_type, bins must be bin edges of the two dimensions.
        For meshgrid bins_type, bins are sent to jnp.histogramdd.
        :param bins_type: binning scheme, can be either irreg or meshgrid.
        """
        self.bins = bins
        self.bins_type = bins_type
        self.needed_parameters = set()

    def simulate_hist(self, *args, **kwargs):
        """Simulate and return hist."""
        raise NotImplementedError

    def implement_binning(self, mc, eff):
        """Apply binning to MC data."""
        if self.bins_type == 'irreg':
            hist = make_hist_irreg_bin_2d(mc, *self.bins, weights=eff)
        elif self.bins_type == 'meshgrid':
            warning = 'The usage of meshgrid binning is highly discouraged.'
            warn(warning)
            hist = make_hist_mesh_grid(mc, bins=self.bins, weights=eff)
        else:
            raise ValueError(f'Unsupported bins_type {self.bins_type}!')
        # as an uncertainty to prevent blowing up
        hist = jnp.clip(hist, 1., jnp.inf)
        return hist

    def get_normalization(self, hist, parameters, batch_size=None):
        """Return the normalization factor of the histogram."""
        if self.norm_type == 'on_pdf':
            normalization_factor = 1 / jnp.sum(hist) * parameters[self.rate_name]
        elif self.norm_type == 'on_sim':
            normalization_factor = 1 / batch_size * parameters[self.rate_name]
        else:
            raise ValueError(f'Unsupported norm_type {self.norm_type}!')
        return normalization_factor

    def deduce(self, *args, **kwargs):
        """Deduce."""
        raise NotImplementedError

    def compile(self):
        """Compile."""
        pass


@export
class ComponentSim(Component):
    """Component that needs MC simulations."""

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
                            dependencies: list = None) -> list:
        """Deduce dependencies."""
        if dependencies is None:
            dependencies = []

        for data_name in data_names:
            # `batch_size` have no dependency
            if data_name == 'batch_size':
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
            if data_name == 'batch_size':
                continue
            dependencies = self.dependencies_deduce(
                data_names=self._plugin_class_registry[data_name].depends_on,
                dependencies=dependencies,
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
                          data_names: list = ('cs1', 'cs2', 'eff'),
                          func_name: str = 'simulate'):
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
            instance = work[0] + self.tag
            code += f'{instance} = {work[0]}()\n'

        # define functions
        code += '\n'
        code += '@partial(jit, static_argnums=(1, ))\n'
        code += f'def {func_name}(key, batch_size, parameters):\n'

        for work in self.worksheet:
            provides = 'key, ' + ', '.join(work[1])
            depends_on = ', '.join(work[2])
            instance = work[0] + self.tag
            code += f'{indent}{provides} = {instance}(key, parameters, {depends_on})\n'
        output = 'key, ' + '[' + ', '.join(data_names) + ']'
        code += f'{indent}return {output}\n'

        self.code = code

        if func_name in _cached_functions.keys():
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
        self._compile = partial(exec, self.code, _cached_functions)

    def deduce(self,
               data_names: list = ('cs1', 'cs2'),
               func_name: str = 'simulate'):
        """Deduce workflow and code."""
        if not isinstance(data_names, (list, tuple)):
            raise ValueError(f'Unsupported data_names type {type(data_names)}!')
        if 'eff' in data_names:
            data_names = list(data_names)
            data_names.remove('eff')
        data_names = list(data_names) + ['eff']

        dependencies = self.dependencies_deduce(data_names)
        self.dependencies_simplify(dependencies)
        self.flush_source_code(data_names, func_name)

    def compile(self):
        """Build simulation function and cache it to share._cached_functions."""
        self._compile()
        self.simulate = _cached_functions[self.func_name]

    def simulate_hist(self,
                      key,
                      batch_size,
                      parameters):
        """Simulate and return histogram."""
        key, result = self.simulate(key, batch_size, parameters)
        mc = result[:-1]
        assert len(mc) == len(self.bins), "Length of bins must be the same as length of bins_on!"
        mc = jnp.asarray(mc).T
        eff = result[-1]  # we guarantee that the last output is efficiency in self.deduce

        hist = self.implement_binning(mc, eff)
        normalization_factor = self.get_normalization(hist, parameters, batch_size)
        hist *= normalization_factor

        return key, hist

    def save_code(self, file_path):
        """Save the code to file."""
        with open(file_path, 'w') as f:
            f.write(self.code)

    def lineage(self, data_name: str = 'cs2'):
        """Return lineage of plugins."""
        assert isinstance(data_name, str)
        pass

    def show_config(self, data_names: list = ('cs1', 'cs2', 'eff')):
        """
        Return configuration options that affect data_names.
        :param data_names: Data type name
        """
        dependencies = self.dependencies_deduce(data_names)
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


@export
class ComponentFixed(Component):
    """Component whose shape is fixed."""

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
        eff = jnp.ones(len(self.data))
        self.hist = self.implement_binning(self.data, eff)
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
