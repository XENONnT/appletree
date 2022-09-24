import appletree
from appletree import Parameter
from appletree import exporter

export, __all__ = exporter()

@export
class Context:
    """
    """
    par_manager = None

    def __init__(self,
                 parameter_config=None,
                 register=None):
        self._plugin_class_registry = dict()
        if register is not None:
            self.register(register)

        self.set_par_manager(parameter_config)
        self.init_parameters()

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
                try:
                    # Looping over the maps of the new plugin and check if
                    # they can be found in the already registered plugins:
                    for new_map, new_items in plugin_class.takes_map.items():
                        if not new_map == map:
                            continue
                        default = items.get_default('0')  # Have to pass will be changed.
                        new_default = new_items.get_default('0')
                        if default == new_default:
                            continue
                        else:
                            mes = (f'Two plugins have a different default value'
                                   f' for the same map. The map'
                                   f' "{new_map}" in "{plugin.__name__}" takes'
                                   f' as a default "{default}"  while in'
                                   f' "{plugin_class.__name__}" the default value'
                                   f' is set to "{new_default}". Please change'
                                   ' one of the defaults.'
                                   )
                            raise ValueError(mes)

                except RuntimeError:
                    # These are option which are inherited from context options.
                    pass

    def new_context(self, *args, **kwargs):
        raise NotImplementedError

    def set_par_manager(self, parameter_config):
        self.par_manager = Parameter(parameter_config)

    def init_parameters(self):
        self.par_manager.init_parameter()

    def set_parameters(self, *args, **kwargs):
        self.par_manager.set_parameter(*args, **kwargs)

    @property
    def parameters(self):
        return self.par_manager._parameter_dict

    def get_parameters(self):
        return self.parameters

    def dependencies_deduce(self):
        pass

    def get_array(self, data_name):
        pass

    def lineage(self, data_name):
        pass

    def get_single_plugin(self, data_name):
        pass
