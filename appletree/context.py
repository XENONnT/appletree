import appletree
from appletree import exporter
from symbol import parameters

export, __all__ = exporter()

@export
class Context:
    """
    """
    par_manager = None
    parameters = dict()

    def __init__(self,
                 parameter_config=None,
                 parameter=None,
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

    def new_context(self,
                    parameter_config=None,
                    parameter=None,
                    register=None):
        raise NotImplementedError

    def init_parameters(self, external_parameter:dict()=None):
        pass

    def set_parameters(self, external_parameter:dict()=None):
        pass

    def update_parameters(self, par_manager):
        if self.par_names == []:
            return

        check, missing = par_manager.check_parameter_exist(self.par_names, return_not_exist=True)
        assert check, "%s not found in par_manager!"%missing

        self.par_values = par_manager.get_parameter(self.par_names)
        self.par_dict = {key : val for key, val in zip(self.par_names, self.par_values)}

        for key, val in zip(self.par_names, self.par_values):
            self.__setattr__(key, val)

    def get_parameters(self, external_parameter:dict()=None):
        pass

    def dependencies_deduce(self):
        pass

    def get_array(self, data_name):
        pass

    def lineage(self, data_name):
        pass

    def get_single_plugin(self, data_name):
        pass
