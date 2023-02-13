from warnings import warn

import numpy as np
from jax import numpy as jnp

from appletree.hist import make_hist_mesh_grid, make_hist_irreg_bin_2d
from appletree.utils import load_data, get_equiprob_bins_2d
from appletree.component import Component, ComponentSim, ComponentFixed
from appletree.randgen import TwoHalfNorm


class Likelihood:
    """Combine all components (e.g. ER, AC, Wall), and calculate log posterior likelihood"""

    def __init__(self, name: str = None, **config):
        """Create an appletree likelihood

        :param config: Dictionary with configuration options that will be applied, should include:

          * data_file_name: the data used in fitting, usually calibration data
          * bins_type: either meshgrid or equiprob
          * bins_on: observables where we will perform inference on, usually [cs1, cs2]
          * x_clip, y_clip: ROI of the fitting, should be list of upper and lower boundary
        """
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self.components = {}
        self._config = config
        self._data_file_name = config['data_file_name']
        self._bins_type = config['bins_type']
        self._bins_on = config['bins_on']
        self._bins = config['bins']
        self._dim = len(self._bins_on)
        if self._dim != 2:
            raise ValueError('Currently only support 2D fitting')
        self.needed_parameters = set()
        self._sanity_check()

        self.data = load_data(self._data_file_name)[self._bins_on].to_numpy()
        mask = (self.data[:, 0] > config['x_clip'][0])
        mask &= (self.data[:, 0] < config['x_clip'][1])
        mask &= (self.data[:, 1] > config['y_clip'][0])
        mask &= (self.data[:, 1] < config['y_clip'][1])
        self.data = self.data[mask]

        if self._bins_type == 'meshgrid':
            warning = f'The usage of meshgrid binning is highly discouraged.'
            warn(warning)
            self.component_bins_type = 'meshgrid'
            x_bins = jnp.linspace(*config['x_clip'], self._bins[0] + 1)
            y_bins = jnp.linspace(*config['y_clip'], self._bins[1] + 1)
            self._bins = (x_bins, y_bins)
            self.data_hist = make_hist_mesh_grid(
                self.data,
                bins=self._bins,
                weights=jnp.ones(len(self.data)),
            )
        elif self._bins_type == 'equiprob':
            if self._dim != 2:
                raise RuntimeError('only 2D equiprob binned likelihood is supported!')
            self._bins = get_equiprob_bins_2d(
                self.data,
                self._bins,
                x_clip=config['x_clip'],
                y_clip=config['y_clip'],
                which_np=jnp)
            self.component_bins_type = 'irreg'
            self.data_hist = make_hist_irreg_bin_2d(
                self.data,
                bins_x=self._bins[0],
                bins_y=self._bins[1],
                weights=jnp.ones(len(self.data)),
            )
        elif self._bins_type == 'irreg':
            if self._dim != 2:
                raise RuntimeError('only 2D irregular binned likelihood is supported!')
            self._bins[0] = jnp.array(self._bins[0])
            self._bins[1] = jnp.array(self._bins[1])
            self.component_bins_type = 'irreg'
            # x-binning should 1 longer than y-binning
            mask0 = len(self._bins[0]) != len(self._bins[1]) + 1
            # all y-binning should have the same length
            mask1 = not all(len(b) == len(self._bins[1][0]) for b in self._bins[1])
            if mask0 or mask1:
                raise ValueError(f'Please check the binning in {self.name}!')
            self.data_hist = make_hist_irreg_bin_2d(
                self.data,
                bins_x=self._bins[0],
                bins_y=self._bins[1],
                weights=jnp.ones(len(self.data)),
            )
        else:
            raise ValueError("'bins_type' should either be meshgrid, equiprob or irreg")

    def __getitem__(self, keys):
        """Get component in likelihood"""
        return self.components[keys]

    def register_component(self,
                           component_cls: Component,
                           component_name: str,
                           file_name: str = None):
        """Create an appletree likelihood.

        :param component_cls: class of Component
        :param component_name: name of Component
        :param file_name: file used in ComponentFixed
        """
        if component_name in self.components:
            raise ValueError(f'Component named {component_name} already existed!')

        # Initialize component
        component = component_cls(
            name=component_name,
            llh_name=self.name,
            bins=self._bins,
            bins_type=self.component_bins_type,
            file_name=file_name,
        )
        component.rate_name = component_name + '_rate'
        kwargs = {'data_names': self._bins_on}
        if isinstance(component, ComponentSim):
            kwargs['func_name'] = self.name + '_' + component_name + '_sim'
            kwargs['data_names'] = self._bins_on + ['eff']
        component.deduce(**kwargs)
        component.compile()

        # Update components sheet
        self.components[component_name] = component

        # Update needed parameters
        self.needed_parameters |= self.components[component_name].needed_parameters

    def get_log_likelihood(self, key, batch_size, parameters):
        """Get log likelihood of given parameters.

        :param key: a pseudo-random number generator (PRNG) key
        :param batch_size: int of number of simulated events
        :param parameters: dict of parameters used in simulation
        """
        key, model_hist = self._simulate_model_hist(key, batch_size, parameters)
        # Poisson likelihood
        llh = jnp.sum(self.data_hist * jnp.log(model_hist) - model_hist)
        llh = float(llh)
        if np.isnan(llh):
            llh = -np.inf
        return key, llh

    def print_likelihood_summary(self,
                                 indent: str = ' '*4,
                                 short: bool = True):
        """Print likelihood summary: components, bins, file names.

        :param indent: str of indent
        :param short: bool, whether only print short summary
        """
        print('\n'+'-'*40)

        print(f'BINNING\n')
        print(f'{indent}bins_type: {self._bins_type}')
        print(f'{indent}bins_on: {self._bins_on}')
        if not short:
            print(f'{indent}bins: {self._bins}')
        print('\n'+'-'*40)

        print(f'DATA\n')
        print(f'{indent}file_name: {self._data_file_name}')
        print(f'{indent}data_rate: {float(self.data_hist.sum())}')
        print('\n'+'-'*40)

        print('MODEL\n')
        for i, component_name in enumerate(self.components):
            name = component_name
            component = self[component_name]
            need = component.needed_parameters

            print(f'{indent}COMPONENT {i}: {name}')
            if isinstance(component, ComponentSim):
                print(f'{indent*2}type: simulation')
                print(f'{indent*2}rate_par: {component.rate_name}')
                print(f'{indent*2}pars: {need}')
                if not short:
                    print(f'{indent*2}worksheet: {component.worksheet}')
            elif isinstance(component, ComponentFixed):
                print(f'{indent*2}type: fixed')
                print(f'{indent*2}file_name: {component._file_name}')
                print(f'{indent*2}rate_par: {component.rate_name}')
                print(f'{indent*2}pars: {need}')
                if not short:
                    print(f'{indent*2}from_file: {component.file_name}')
            else:
                pass
            print()

        print('-'*40)

    def _sanity_check(self):
        """Check equality between number of bins group and observables"""
        if len(self._bins_on) != len(self._bins):
            raise RuntimeError('Length of bins must be the same as length of bins_on!')

    def _simulate_model_hist(self, key, batch_size, parameters):
        """Histogram of simulated observables.

        :param key: a pseudo-random number generator (PRNG) key
        :param batch_size: int of number of simulated events
        :param parameters: dict of parameters used in simulation
        """
        hist = jnp.zeros_like(self.data_hist)
        for component_name, component in self.components.items():
            if isinstance(component, ComponentSim):
                key, _hist = component.simulate_hist(key, batch_size, parameters)
            elif isinstance(component, ComponentFixed):
                _hist = component.simulate_hist(parameters)
            else:
                raise TypeError(f'unsupported component type for {component_name}!')
            hist += _hist
        return key, hist

    def simulate_weighed_data(self, key, batch_size, parameters):
        result = []
        for component_name, component in self.components.items():
            if isinstance(component, ComponentSim):
                key, _result = component.simulate_weighed_data(key, batch_size, parameters)
            elif isinstance(component, ComponentFixed):
                _result = component.simulate_weighed_data(parameters)
            else:
                raise TypeError(f'unsupported component type for {component_name}!')
            result.append(_result)
        result = [r for r in np.hstack(result)]
        return key, result


class LikelihoodLit(Likelihood):
    """Using literature constraint to build LLH"""

    def __init__(self, name: str = None, **config):
        """Create an appletree likelihood

        :param config: Dictionary with configuration options that will be applied, should include:
        """
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self.components = {}
        self._config = config
        self._bins = None
        self._bins_type = None
        self._bins_on = config['bins_on']
        self._dim = len(self._bins_on)

        self.needed_parameters = set()
        self.component_bins_type = None
        logpdf_args = self._config['logpdf_args']
        self.logpdf_args = {
            k: np.array(v) for k, v in zip(*logpdf_args)}

        self.variable_type = config['variable_type']
        self._sanity_check()

        if self.variable_type == 'twohalfnorm':
            setattr(self, 'logpdf', lambda x: TwoHalfNorm.logpdf(
                x=x, **self.logpdf_args))
        else:
            raise NotImplementedError

        self.warning = 'Currently only support one dimensional inference'

    def _sanity_check(self):
        """Check sanities of supported distribution and dimension"""
        if self.variable_type != 'twohalfnorm':
            raise RuntimeError('Currently only twohalfnorm is supported')
        if self._dim != 1:
            raise AssertionError(self.warning)

    def get_log_likelihood(self, key, batch_size, parameters):
        """Get log likelihood of given parameters.

        :param key: a pseudo-random number generator (PRNG) key
        :param batch_size: int of number of simulated events
        :param parameters: dict of parameters used in simulation
        """
        key, result = self._simulate_yields(key, batch_size, parameters)
        yields, eff = result
        llh = self.logpdf(yields)
        llh = (llh * eff).sum()
        llh = float(llh)
        if np.isnan(llh):
            llh = -np.inf
        return key, llh

    def _simulate_yields(self, key, batch_size, parameters):
        """Histogram of simulated observables.

        :param key: a pseudo-random number generator (PRNG) key
        :param batch_size: int of number of simulated events
        :param parameters: dict of parameters used in simulation
        """
        if len(self.components) != 1:
            raise AssertionError(self.warning)
        component = list(self.components.keys())[0]
        key, result = self.components[component].simulate(
            key, batch_size, parameters)
        # Move data to CPU
        result = [np.array(r) for r in result]
        return key, result

    def print_likelihood_summary(self,
                                 indent: str = ' '*4,
                                 short: bool = True):
        """Print likelihood summary: components, bins, file names.

        :param indent: str of indent
        :param short: bool, whether only print short summary
        """
        print('\n'+'-'*40)

        print(f'BINNING\n')
        print(f'{indent}variable_type: {self.variable_type}')
        print(f'{indent}variable: {self._bins_on}')
        print('\n'+'-'*40)

        print(f'LOGPDF\n')
        print(f'{indent}logpdf_args:')
        for k, v in self.logpdf_args.items():
            print(f'{indent*2}{k}: {v}')
        print('\n'+'-'*40)

        print('MODEL\n')
        for i, component_name in enumerate(self.components):
            name = component_name
            component = self[component_name]
            need = component.needed_parameters

            print(f'{indent}COMPONENT {i}: {name}')
            if isinstance(component, ComponentSim):
                print(f'{indent*2}type: simulation')
                print(f'{indent*2}rate_par: {component.rate_name}')
                print(f'{indent*2}pars: {need}')
                if not short:
                    print(f'{indent*2}worksheet: {component.worksheet}')
            elif isinstance(component, ComponentFixed):
                print(f'{indent*2}type: fixed')
                print(f'{indent*2}file_name: {component._file_name}')
                print(f'{indent*2}rate_par: {component.rate_name}')
                print(f'{indent*2}pars: {need}')
                if not short:
                    print(f'{indent*2}from_file: {component.file_name}')
            else:
                pass
            print()

        print('-'*40)
