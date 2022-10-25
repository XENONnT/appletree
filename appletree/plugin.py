import inspect

from immutabledict import immutabledict

from appletree.utils import exporter

export, __all__ = exporter()


@export
class Plugin():
    """Plugin, which is the smallest simulation unit."""

    # the plugin's dependency(the arguments of `simulate`)
    depends_on = []

    # the plugin can provide(`simulate` will return)
    provides = []

    # relevant parameters, will be fitted in MCMC
    parameters = ()

    # Set using the takes_config decorator
    takes_config = immutabledict()

    def __init__(self):
        """Initialization."""
        if not self.depends_on:
            raise ValueError('depends_on not provided for '
                             f'{self.__class__.__name__}')

        if not self.provides:
            raise ValueError('provides not provided for '
                             f'{self.__class__.__name__}')

        # configs are loaded when a plugin is initialized
        for config in self.takes_config.values():
            config.build()

        self.sanity_check()

    def __call__(self, *args, **kwargs):
        """Calls self.simulate"""
        return self.simulate(*args, **kwargs)

    def simulate(self, *args, **kwargs):
        """Simulate."""
        raise NotImplementedError

    def sanity_check(self):
        """Check the consistency between `depends_on`, `provides` and in(out)put of `simulation`"""
        arguments = inspect.getfullargspec(self.simulate)[0]
        assert arguments[1] == 'key' and arguments[2] == 'parameters'
        for i, depend in enumerate(self.depends_on, start=3):
            assert arguments[i] == depend, f'Plugin {self.__class__.__name__} is insane, check dependency!'


@export
def add_plugin_extensions(module1, module2):
    """Add plugins of module2 to module1"""
    for x in dir(module2):
        x = getattr(module2, x)
        if not isinstance(x, type(type)):
            continue
        _add_plugin_extension(module1, x)


@export
def _add_plugin_extension(module, plugin):
    """Add plugin to module"""
    if issubclass(plugin, Plugin):
        setattr(module, plugin.__name__, plugin)
