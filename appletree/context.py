import os
import inspect
from functools import partial

import appletree
from appletree import plugins
from appletree.plugin import Plugin
from appletree.parameter import Parameter
from appletree.utils import exporter

export, __all__ = exporter()

PARPATH = os.path.join(os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe()))), 'parameters')

__all__ += 'PARPATH'.split()

@export
class Context:
    """
    """
    code: str = None
    old_code: str = None
    tag = '_'  # for instance name of the plugins
    initialized_names = []

    def __init__(self,
                 register=None):
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
                            data_names:list=['cs1', 'cs2', 'eff'], 
                            dependencies:list=None) -> list:
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
        self.needed_parameters = []
        for plugin in dependencies[::-1]:
            plugin = plugin['plugin']
            if plugin.__name__ in already_seen:
                continue
            self.worksheet.append([plugin.__name__, plugin.provides, plugin.depends_on])
            already_seen.append(plugin.__name__)
            self.needed_parameters += plugin.parameters
        # filter out duplicated parameters
        self.needed_parameters = list(set(self.needed_parameters))
        self.needed_parameters.sort()

    def deduce(self, 
               data_names:list=['cs1', 'cs2', 'eff'], 
               func_name:str='simulate',
               seed=None):
        dependencies = self.dependencies_deduce(data_names)
        self.dependencies_simplify(dependencies)
        self.flush_source_code(data_names, func_name)

    def flush_source_code(self, 
                          data_names:list=['cs1', 'cs2', 'eff'],
                          func_name:str='simulate'):
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

    @property
    def code(self):
        return self._code

    @code.setter
    def code(self, code):
        self._code = code
        self.compile = partial(exec, self.code)

    def save_code(self, file_path):
        with open(file_path, 'w') as f:
            f.write(self.code)

    def lineage(self, data_name:str='cs2'):
        assert isinstance(data_name, str)
        pass


@export
class ERBand(Context):
    def __init__(self):
        super().__init__()

        self.register_all(plugins)
