from warnings import warn
from typing import Type, Dict, Set, Optional, cast
import inspect
from copy import deepcopy

import numpy as np
from jax import numpy as jnp

from scipy.stats import norm

from appletree import randgen
from appletree.hist import make_hist_mesh_grid, make_hist_irreg_bin_1d, make_hist_irreg_bin_2d
from appletree.utils import load_data, get_equiprob_bins_1d, get_equiprob_bins_2d
from appletree.component import Component, ComponentSim, ComponentFixed
from appletree.randgen import TwoHalfNorm, BandTwoHalfNorm


def need_replacing_alias(func):
    """Decorator to replace alias in parameters from key to value."""

    def wrapper(self, *args, **kwargs):
        sig = inspect.signature(func)
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()

        # Check if 'parameters' is in the arguments and replace aliases
        if "parameters" in bound.arguments:
            # Replace alias in 'parameters'
            bound.arguments["parameters"] = self.replace_alias(bound.arguments["parameters"])
        else:
            raise ValueError(f"'parameters' must be in the arguments of {func.__name__}!")

        # Call the function with possibly modified arguments
        return func(*bound.args, **bound.kwargs)

    return wrapper


class Likelihood:
    """Combine all components (e.g. ER, AC, Wall), and calculate log posterior likelihood."""

    def __init__(self, name: Optional[str] = None, **config):
        """Create an appletree likelihood.

        Args:
            config: Dictionary with configuration options that will be applied, should include:
                * data_file_name: the data used in fitting, usually calibration data
                * bins_type: either meshgrid or equiprob
                * bins_on: observables where we will perform inference on, usually [cs1, cs2]
                * x_clip, y_clip: ROI of the fitting, should be list of upper and lower boundary
                * parameter_alias: alias of parameters that will be renamed from key to value

        """
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self.components = cast(Dict[str, Component], dict())
        self._config = config
        self._data_file_name = config["data_file_name"]
        self._bins_type = config["bins_type"]
        self._bins_on = config["bins_on"]
        self._bins = config["bins"]
        self._parameter_alias = config.get("parameter_alias", dict())
        if isinstance(self._bins_on, str):
            self._dim = 1
            self._bins_on = [self._bins_on]
            if isinstance(self._bins, int):
                self._bins = [self._bins]
        elif isinstance(self._bins_on, list):
            self._dim = len(self._bins_on)
            assert isinstance(
                self._bins, list
            ), "bins should be list if not 1D fitting, but got {self._bins}!"
        else:
            raise ValueError(
                f"bins_on should be either str or list of str, but got {self._bins_on}."
            )
        self.needed_parameters: Set[str] = set()
        self._sanity_check()

        self.data = load_data(self._data_file_name)[self._bins_on].to_numpy()
        if self._dim == 1:
            mask = self.data[:, 0] > config["clip"][0]
            mask &= self.data[:, 0] < config["clip"][1]
        else:
            mask = self.data[:, 0] > config["x_clip"][0]
            mask &= self.data[:, 0] < config["x_clip"][1]
            mask &= self.data[:, 1] > config["y_clip"][0]
            mask &= self.data[:, 1] < config["y_clip"][1]
        self.data = self.data[mask]

        self.set_binning(config)

    def __getitem__(self, keys):
        """Get component in likelihood."""
        return self.components[keys]

    def set_binning(self, config):
        """Set binning of likelihood."""
        if self._dim == 1 or self._dim == 2:
            pass
        else:
            raise ValueError(f"Currently only support 1D and 2D, but got {self._dim}D!")
        if self._bins_type == "meshgrid":
            warning = "The usage of meshgrid binning is highly discouraged."
            warn(warning)
            self.component_bins_type = "meshgrid"
            if self._dim == 1:
                if isinstance(self._bins[0], int):
                    bins = jnp.linspace(*config["clip"], self._bins[0] + 1)
                else:
                    bins = jnp.array(self._bins[0])
                    if "x_clip" in config:
                        warning = "x_clip is ignored when bins_type is meshgrid and bins is not int"
                        warn(warning)
                self._bins = (bins,)
            elif self._dim == 2:
                if isinstance(self._bins[0], int):
                    x_bins = jnp.linspace(*config["x_clip"], self._bins[0] + 1)
                else:
                    x_bins = jnp.array(self._bins[0])
                    if "x_clip" in config:
                        warning = "x_clip is ignored when bins_type is meshgrid and bins is not int"
                        warn(warning)
                if isinstance(self._bins[1], int):
                    y_bins = jnp.linspace(*config["y_clip"], self._bins[1] + 1)
                else:
                    y_bins = jnp.array(self._bins[1])
                    if "y_clip" in config:
                        warning = "y_clip is ignored when bins_type is meshgrid and bins is not int"
                        warn(warning)
                self._bins = (x_bins, y_bins)
            self.data_hist = make_hist_mesh_grid(
                self.data,
                bins=self._bins,
                weights=jnp.ones(len(self.data)),
            )
        elif self._bins_type == "equiprob":
            if not all([isinstance(b, int) for b in self._bins]):
                raise RuntimeError("bins can only be int if bins_type is equiprob")
            if self._dim == 1:
                self._bins = get_equiprob_bins_1d(
                    self.data[:, 0],
                    self._bins[0],
                    clip=config["clip"],
                    which_np=jnp,
                )
                self._bins = [self._bins]
                self.data_hist = make_hist_irreg_bin_1d(
                    self.data[:, 0],
                    bins=self._bins[0],
                    weights=jnp.ones(len(self.data)),
                )
            elif self._dim == 2:
                self._bins = get_equiprob_bins_2d(
                    self.data,
                    self._bins,
                    x_clip=config["x_clip"],
                    y_clip=config["y_clip"],
                    which_np=jnp,
                )
                self.data_hist = make_hist_irreg_bin_2d(
                    self.data,
                    bins_x=self._bins[0],
                    bins_y=self._bins[1],
                    weights=jnp.ones(len(self.data)),
                )
            self.component_bins_type = "irreg"
        elif self._bins_type == "irreg":
            self._bins = [jnp.array(b) for b in self._bins]
            if self._dim == 1:
                if isinstance(self._bins[0], int):
                    bins = jnp.linspace(*config["clip"], self._bins[0] + 1)
                else:
                    bins = jnp.array(self._bins[0])
                    if "x_clip" in config:
                        warning = "x_clip is ignored when bins_type is meshgrid and bins is not int"
                        warn(warning)
                self._bins = (bins,)
            elif self._dim == 2:
                if isinstance(self._bins[0], int):
                    x_bins = jnp.linspace(*config["x_clip"], self._bins[0] + 1)
                else:
                    x_bins = jnp.array(self._bins[0])
                    if "x_clip" in config:
                        warning = "x_clip is ignored when bins_type is meshgrid and bins is not int"
                        warn(warning)
                if isinstance(self._bins[1], int):
                    y_bins = jnp.linspace(*config["y_clip"], self._bins[1] + 1)
                else:
                    y_bins = jnp.array(self._bins[1])
                    if "y_clip" in config:
                        warning = "y_clip is ignored when bins_type is meshgrid and bins is not int"
                        warn(warning)
                self._bins = (x_bins, y_bins)
            self.component_bins_type = "irreg"
            if self._dim == 2:
                mask = len(self._bins[0]) != len(self._bins[1]) + 1
                if mask:
                    raise ValueError(
                        f"The x-binning should 1 longer than y-binning, "
                        f"please check the binning in {self.name}!"
                    )
                mask = not all(len(b) == len(self._bins[1][0]) for b in self._bins[1])
                if mask:
                    raise ValueError(
                        f"All y-binning should have the same length, "
                        f"please check the binning in {self.name}!"
                    )
            if self._dim == 1:
                self.data_hist = make_hist_irreg_bin_1d(
                    self.data[:, 0],
                    bins=self._bins[0],
                    weights=jnp.ones(len(self.data)),
                )
            elif self._dim == 2:
                self.data_hist = make_hist_irreg_bin_2d(
                    self.data,
                    bins_x=self._bins[0],
                    bins_y=self._bins[1],
                    weights=jnp.ones(len(self.data)),
                )
        else:
            raise ValueError("'bins_type' should either be meshgrid, equiprob or irreg")

    def register_component(
        self, component_cls: Type[Component], component_name: str, file_name: Optional[str] = None
    ):
        """Create an appletree likelihood.

        Args:
            component_cls: class of Component.
            component_name: name of Component.
            file_name: file used in ComponentFixed.

        """
        if component_name in self.components:
            raise ValueError(f"Component named {component_name} already existed!")

        # Initialize component
        component = component_cls(
            name=component_name,
            llh_name=self.name,
            bins=self._bins,
            bins_type=self.component_bins_type,
            file_name=file_name,
        )
        component.rate_name = component_name + "_rate"
        kwargs = {"data_names": self._bins_on}
        if isinstance(component, ComponentSim):
            kwargs["func_name"] = self.name + "_" + component_name + "_sim"
            kwargs["data_names"] = self._bins_on + ["eff"]
        component.deduce(**kwargs)
        component.compile()

        # Update components sheet
        self.components[component_name] = component

        # Update needed parameters
        self.needed_parameters |= self.components[component_name].needed_parameters

        # Replace alias in needed parameters
        for k, v in self._parameter_alias.items():
            if v in self.needed_parameters:
                self.needed_parameters.add(k)
                self.needed_parameters.remove(v)

    def replace_alias(self, parameters):
        """Replace alias in parameters from key to value.

        Note that the value will be popped out.

        """
        _parameters = deepcopy(parameters)
        for k, v in self._parameter_alias.items():
            if k in _parameters:
                _parameters[v] = _parameters.pop(k)
        return _parameters

    def _sanity_check(self):
        """Check equality between number of bins group and observables."""
        if len(self._bins_on) != len(self._bins):
            raise RuntimeError("Length of bins must be the same as length of bins_on!")

    @need_replacing_alias
    def _simulate_model_hist(self, key, batch_size, parameters):
        """Histogram of simulated observables.

        Args:
            key: a pseudo-random number generator (PRNG) key.
            batch_size: int of number of simulated events.
            parameters: dict of parameters used in simulation.

        """
        hist = jnp.zeros_like(self.data_hist)
        for component_name, component in self.components.items():
            if isinstance(component, ComponentSim):
                key, _hist = component.simulate_hist(key, batch_size, parameters)
            elif isinstance(component, ComponentFixed):
                _hist = component.simulate_hist(parameters)
            else:
                raise TypeError(f"unsupported component type for {component_name}!")
            hist += _hist
        return key, hist

    @need_replacing_alias
    def simulate_weighted_data(self, key, batch_size, parameters):
        """Simulate weighted histogram.

        Args:
            key: a pseudo-random number generator (PRNG) key.
            batch_size: int of number of simulated events.
            parameters: dict of parameters used in simulation.

        """
        result = []
        for component_name, component in self.components.items():
            if isinstance(component, ComponentSim):
                key, _result = component.simulate_weighted_data(key, batch_size, parameters)
            elif isinstance(component, ComponentFixed):
                _result = component.simulate_weighted_data(parameters)
            else:
                raise TypeError(f"unsupported component type for {component_name}!")
            result.append(_result)
        result = list(r for r in np.hstack(result))
        return key, result

    @need_replacing_alias
    def get_log_likelihood(self, key, batch_size, parameters):
        """Get log likelihood of given parameters.

        Args:
            key: a pseudo-random number generator (PRNG) key.
            batch_size: int of number of simulated events.
            parameters: dict of parameters used in simulation.

        """
        key, model_hist = self._simulate_model_hist(key, batch_size, parameters)
        # Poisson likelihood
        llh = jnp.sum(self.data_hist * jnp.log(model_hist) - model_hist)
        llh = float(llh)
        if np.isnan(llh):
            llh = -np.inf
        return key, llh

    @need_replacing_alias
    def get_num_events_accepted(self, batch_size, parameters):
        """Get number of events in the histogram under given parameters.

        Args:
            batch_size: int of number of simulated events.
            parameters: dict of parameters used in simulation.

        """
        key = randgen.get_key()
        _, model_hist = self._simulate_model_hist(key, batch_size, parameters)
        return model_hist.sum()

    def print_likelihood_summary(self, indent: str = " " * 4, short: bool = True):
        """Print likelihood summary: components, bins, file names.

        Args:
            indent: str of indent.
            short: bool, whether only print short summary.
        """
        print("\n" + "-" * 40)

        print("BINNING\n")
        print(f"{indent}bins_type: {self._bins_type}")
        print(f"{indent}bins_on: {self._bins_on}")
        if not short:
            print(f"{indent}bins: {self._bins}")
        print("\n" + "-" * 40)

        print("DATA\n")
        print(f"{indent}file_name: {self._data_file_name}")
        print(f"{indent}data_rate: {float(self.data_hist.sum())}")
        print("\n" + "-" * 40)

        print("MODEL\n")
        for i, component_name in enumerate(self.components):
            name = component_name
            component = self[component_name]
            need = component.needed_parameters

            print(f"{indent}COMPONENT {i}: {name}")
            if isinstance(component, ComponentSim):
                print(f"{indent * 2}type: simulation")
                print(f"{indent * 2}rate_par: {component.rate_name}")
                print(f"{indent * 2}pars: {need}")
                if not short:
                    print(f"{indent * 2}worksheet: {component.worksheet}")
            elif isinstance(component, ComponentFixed):
                print(f"{indent * 2}type: fixed")
                print(f"{indent * 2}file_name: {component._file_name}")
                print(f"{indent * 2}rate_par: {component.rate_name}")
                print(f"{indent * 2}pars: {need}")
                if not short:
                    print(f"{indent * 2}from_file: {component._file_name}")
            else:
                pass
            print()

        print("-" * 40)


class LikelihoodLit(Likelihood):
    """Using literature constraint to build LLH.

    The idea is to simulate light and charge yields directly with given energy distribution. And
    then fit the result with provided literature measurement points. The energy distribution will
    always be twohalfnorm(TwoHalfNorm), norm or band, which is specified by

    """

    def __init__(self, name: Optional[str] = None, **config):
        """Create an appletree likelihood.

        Args:
            config: Dictionary with configuration options that will be applied.

        """
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self.components = dict()
        self._config = config
        self._bins = None
        self._bins_type = None
        self._bins_on = config["bins_on"]
        self._dim = len(self._bins_on)
        self._parameter_alias = config.get("parameter_alias", dict())

        self.needed_parameters: Set[str] = set()
        self.component_bins_type = None
        logpdf_args = self._config["logpdf_args"]
        self.logpdf_args = {k: np.array(v) for k, v in zip(*logpdf_args)}

        self.variable_type = config["variable_type"]
        self.warning = "Currently only support one dimensional inference"
        self._sanity_check()

        if self.variable_type == "twohalfnorm":
            setattr(self, "logpdf", lambda x, y: TwoHalfNorm.logpdf(x=y, **self.logpdf_args))
        elif self.variable_type == "norm":
            setattr(self, "logpdf", lambda x, y: norm.logpdf(x=y, **self.logpdf_args))
        elif self.variable_type == "band":
            self.bandtwohalfnorm = BandTwoHalfNorm(**self.logpdf_args)
            setattr(self, "logpdf", lambda x, y: self.bandtwohalfnorm.logpdf(x=x, y=y))
        else:
            raise NotImplementedError

    def _sanity_check(self):
        """Check sanities of supported distribution and dimension."""
        if self.variable_type not in ["twohalfnorm", "norm", "band"]:
            raise RuntimeError("Currently only twohalfnorm, norm and band are supported")
        if self._dim != 2:
            raise AssertionError(self.warning)

    @need_replacing_alias
    def _simulate_yields(self, key, batch_size, parameters):
        """Histogram of simulated observables.

        Args:
            key: a pseudo-random number generator (PRNG) key.
            batch_size: int of number of simulated events.
            parameters: dict of parameters used in simulation.

        """
        if len(self.components) != 1:
            raise AssertionError(self.warning)
        key, result = self.components[self.only_component].simulate(key, batch_size, parameters)
        # Move data to CPU
        result = [np.array(r) for r in result]
        return key, result

    def register_component(self, *args, **kwargs):
        if len(self.components) != 0:
            raise AssertionError(self.warning)
        super().register_component(*args, **kwargs)
        # cache the component name
        self.only_component = list(self.components.keys())[0]

    @need_replacing_alias
    def get_log_likelihood(self, key, batch_size, parameters):
        """Get log likelihood of given parameters.

        Args:
            key: a pseudo-random number generator (PRNG) key.
            batch_size: int of number of simulated events.
            parameters: dict of parameters used in simulation.

        """
        if batch_size != 1:
            warning = (
                "You specified the batch_size larger than 1, "
                "but it should and will be changed to 1 in literature fitting!"
            )
            warn(warning)
        key, result = self._simulate_yields(key, 1, parameters)
        energies, yields, eff = result
        llh = self.logpdf(energies, yields)
        llh = (llh * eff).sum()
        llh = float(llh)
        if np.isnan(llh):
            llh = -np.inf
        return key, llh

    def print_likelihood_summary(self, indent: str = " " * 4, short: bool = True):
        """Print likelihood summary: components, bins, file names.

        Args:
            indent: str of indent.
            short: bool, whether only print short summary.
        """
        print("\n" + "-" * 40)

        print("BINNING\n")
        print(f"{indent}variable_type: {self.variable_type}")
        print(f"{indent}variable: {self._bins_on}")
        print("\n" + "-" * 40)

        print("LOGPDF\n")
        print(f"{indent}logpdf_args:")
        for k, v in self.logpdf_args.items():
            print(f"{indent * 2}{k}: {v}")
        print("\n" + "-" * 40)

        print("MODEL\n")
        for i, component_name in enumerate(self.components):
            name = component_name
            component = self[component_name]
            need = component.needed_parameters

            print(f"{indent}COMPONENT {i}: {name}")
            if isinstance(component, ComponentSim):
                print(f"{indent * 2}type: simulation")
                print(f"{indent * 2}rate_par: {component.rate_name}")
                print(f"{indent * 2}pars: {need}")
                if not short:
                    print(f"{indent * 2}worksheet: {component.worksheet}")
            elif isinstance(component, ComponentFixed):
                print(f"{indent * 2}type: fixed")
                print(f"{indent * 2}file_name: {component._file_name}")
                print(f"{indent * 2}rate_par: {component.rate_name}")
                print(f"{indent * 2}pars: {need}")
                if not short:
                    print(f"{indent * 2}from_file: {component._file_name}")
            else:
                pass
            print()

        print("-" * 40)
