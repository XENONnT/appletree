from warnings import warn
from functools import partial

import jax.numpy as jnp

import appletree
from appletree.plugin import Plugin
from appletree.share import cached_functions
from appletree.utils import exporter, load_data
from appletree.hist import make_hist_mesh_grid, make_hist_irreg_bin_2d

export, __all__ = exporter()

@export
class Component:
    rate_name: str = ''
    norm_type: str = ''
    tag: str = '_'  # for instance name of the plugins

    def __init__(self, 
                 bins:list=[], 
                 bins_type:str=''):
        self.bins = bins
        self.bins_type = bins_type
        self.needed_parameters = set()

    def simulate_hist(self, *args, **kwargs):
        raise NotImplementedError

    def implement_binning(self, mc, eff):
        if self.bins_type == 'irreg':
            hist = make_hist_irreg_bin_2d(mc, self.bins[0], self.bins[1], weights=eff)
        elif self.bins_type == 'meshgrid':
            warning = f'The usage of meshgrid binning is highly discouraged.'
            warn(warning)
            hist = make_hist_mesh_grid(mc, bins=self.bins, weights=eff)
        else:
            raise ValueError(f'Unsupported bins_type {self.bins_type}!')
        hist = jnp.clip(hist, 1., jnp.inf) # as an uncertainty to prevent blowing up
        return hist

    def get_normalization(self, hist, parameters, batch_size=None):
        if self.norm_type == 'on_pdf':
            normalization_factor = 1 / jnp.sum(hist) * parameters[self.rate_name]
        elif self.norm_type == 'on_sim':
            normalization_factor = 1 / batch_size * parameters[self.rate_name]
        else:
            raise ValueError(f'Unsupported norm_type {self.norm_type}!')
        return normalization_factor

    def deduce(self, *args, **kwargs):
        raise NotImplementedError

    def compile(self):
        pass

@export
class ComponentSim(Component):
    """
    """
    code: str = None
    old_code: str = None

    def __init__(self,
                 register = None, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._plugin_class_registry = dict()
        if register is not None:
            self.register(register)

    def register(self, plugin_class):
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

            for map, items in plugin.takes_map.items():
                # Looping over the maps of the new plugin and check if
                # they can be found in the already registered plugins:
                for new_map, new_items in plugin_class.takes_map.items():
                    if not new_map == map:
                        continue
                    if items.file_name == new_items.file_name:
                        continue
                    else:
                        mes = (f'Two plugins have a different file name'
                                f' for the same map. The map'
                                f' "{new_map}" in "{plugin.__name__}" takes'
                                f' the file name as "{new_items.file_name}"  while in'
                                f' "{plugin_class.__name__}" the file name'
                                f' is set to "{items.file_name}". Please change'
                                ' one of the file names.'
                                )
                        raise ValueError(mes)

    def register_all(self, module):
        """
        Register all plugins defined in module.
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
                            data_names: list = ['cs1', 'cs2', 'eff'], 
                            dependencies: list = None) -> list:
        if dependencies is None:
            dependencies = []

        for data_name in data_names:
            # `batch_size` have no dependency
            if data_name == 'batch_size':
                continue
            try:
                dependencies.append({'plugin': self._plugin_class_registry[data_name], 
                                     'provides': data_name, 
                                     'depends_on': self._plugin_class_registry[data_name].depends_on})
            except:
                raise ValueError(f'Can not find dependency for {data_name}')

        for data_name in data_names:
            # `batch_size` has no dependency
            if data_name == 'batch_size':
                continue
            dependencies = self.dependencies_deduce(data_names=self._plugin_class_registry[data_name].depends_on, dependencies=dependencies)

        return dependencies

    def dependencies_simplify(self, dependencies):
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
                          data_names:list=['cs1', 'cs2', 'eff'],
                          func_name:str='simulate'):
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

        if func_name in cached_functions.keys():
            warning = f'Function name {func_name} is already cached. Running compile() will overwrite it.'
            warn(warning)

    @property
    def code(self):
        return self._code

    @code.setter
    def code(self, code):
        self._code = code
        self._compile = partial(exec, self.code, cached_functions)

    def deduce(self, 
               data_names:list=['cs1', 'cs2'], 
               func_name:str='simulate'):
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
        self._compile()
        self.simulate = cached_functions[self.func_name]

    def simulate_hist(self, 
                      key, 
                      batch_size, 
                      parameters):
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
        with open(file_path, 'w') as f:
            f.write(self.code)

    def lineage(self, data_name: str = 'cs2'):
        assert isinstance(data_name, str)
        pass

@export
class ComponentFixed(Component):
    file_name: str = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def deduce(self, 
               data_names:list=['cs1', 'cs2']):
        self.data = load_data(self.file_name)[data_names].to_numpy()
        self.hist = self.implement_binning(self.data, jnp.ones(len(self.data)))
        self.needed_parameters.add(self.rate_name)

    def simulate(self):
        raise NotImplementedError

    def simulate_hist(self, 
                      parameters, 
                      *args, **kwargs):
        normalization_factor = self.get_normalization(self.hist, parameters, len(self.data))
        return self.hist * normalization_factor
