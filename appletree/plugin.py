import inspect
from copy import deepcopy
from typing import List, Tuple, Optional

from immutabledict import immutabledict

from appletree import utils
from appletree.utils import exporter

export, __all__ = exporter()


@export
class Plugin:
    """The smallest simulation unit."""

    # Do not initialize this class because it is base
    __is_base = True

    # the plugin's dependency(the arguments of `simulate`)
    depends_on: List[str] = []

    # the plugin can provide(`simulate` will return)
    provides: List[str] = []

    # relevant parameters, will be fitted in MCMC
    parameters: Tuple = ()

    # Set using the takes_config decorator
    takes_config = immutabledict()

    def __init__(self, llh_name: Optional[str] = None):
        """Initialization."""
        # llh_name will tell us which map to use
        self.llh_name = llh_name
        if not self.depends_on:
            raise ValueError(f"depends_on not provided for {self.__class__.__name__}")

        if not self.provides:
            raise ValueError(f"provides not provided for {self.__class__.__name__}")

        # configs are loaded when a plugin is initialized
        for config in self.takes_config.values():
            config.build(self.llh_name)

        self.sanity_check()

        # Do not set configurations as static! This is very important!!!
        for config in self.takes_config.values():
            setattr(self, config.name, deepcopy(config))

    def __call__(self, *args, **kwargs):
        """Calls self.simulate."""
        return self.simulate(*args, **kwargs)

    def simulate(self, *args, **kwargs):
        """The main simulation function.

        Args:
            key: a jnp.array with length 2, used to generate random variables.
                See https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html
            parameters: a dictionary with key being parameters' names. Plugin will
                get values of self.parameters from this dictionary.
            args: other args following ``key`` and ``parameters`` must be in the order of
                self.depends_on.

        Returns:
            ``key`` and output simulated variables, ordered by self.provides. ``key`` will
            be updated if it's used inside self.simulate to generate random variables.

        """
        raise NotImplementedError

    def sanity_check(self):
        """Check the consistency between ``depends_on``, ``provides`` and in(out)put of
        ``self.simulate``"""
        arguments = inspect.getfullargspec(self.simulate)[0]
        if arguments[1] != "key":
            mesg = f"First argument of {self.__class__.__name__}"
            mesg += ".simulate should be 'key'."
            raise ValueError(mesg)
        if arguments[2] != "parameters":
            mesg = f"Second argument of {self.__class__.__name__}"
            mesg += ".simulate should be 'parameters'."
            raise ValueError(mesg)
        for i, depend in enumerate(self.depends_on, start=3):
            if arguments[i] != depend:
                mesg = f"{i}th argument of {self.__class__.__name__}"
                mesg += f".simulate should be '{depend}'. "
                mesg += f"Plugin {self.__class__.__name__} is insane, check dependency!"
                raise ValueError(mesg)


@export
def add_plugin_extensions(module1, module2, force=False):
    """Add plugins of module2 to module1."""
    utils.add_extensions(module1, module2, Plugin, force=force)


@export
def _add_plugin_extension(module, plugin, force=False):
    """Add plugin to module."""
    utils._add_extension(module, plugin, Plugin, force=force)
