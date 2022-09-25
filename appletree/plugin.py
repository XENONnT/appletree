import inspect

from immutabledict import immutabledict

from appletree.utils import exporter

export, __all__ = exporter()

@export
class Plugin():
    # the plugin's dependency(the arguments of `simulate`)
    depends_on = []

    # the plugin can provide(`simulate` will return)
    provides = []

    # relevant parameters, will be fitted in MCMC
    parameters = tuple()

    # Set using the takes_map decorator
    takes_map = immutabledict()

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.simulate(*args, **kwargs)

    def simulate(self, *args, **kwargs):
        raise NotImplementedError

    def sanity_check(self):
        """
        Check the consistency between `depends_on`, `provides` and in(out)put of `simulation`
        """
        arguments = inspect.getfullargspec(self.simulate)[0]
        assert arguments[1] == 'key' and arguments[1] == 'parameters'
        for i, depend in enumerate(self.depends_on, start=2):
            assert arguments[i] == depend, f'Plugin {self.__name__} is insane, check dependency!'
