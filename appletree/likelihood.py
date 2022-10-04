from warnings import warn

import numpy as np
from jax import numpy as jnp

from appletree.hist import make_hist_mesh_grid, make_hist_irreg_bin_2d
from appletree.utils import load_data, get_equiprob_bins_2d
from appletree.component import Component, ComponentSim, ComponentFixed


class Likelihood:
    """Combine all components(e.g. ER, AC, Wall), and calculate log posterior likelihood"""

    def __init__(self, config: dict):
        """
        Create an appletree likelihood
        :param config: Dictionary with configuration options that will be applied, should include:
        - data_file_name: the data used in fitting, usually calibration data
        - bins_type: either meshgrid or equiprob
        - bins_on: observables where we will perform inference on, usually [cs1, cs2]
        - x_clip, y_clip: ROI of the fitting, should be list of upper and lower boundary
        """
        self.components = {}
        self._config = config
        self.data_file_name = config['data_file_name']
        self.bins_type = config['bins_type']
        self.bins_on = config['bins_on']
        self.bins = config['bins']
        self.dim = len(self.bins_on)
        if self.dim != 2:
            raise ValueError('Currently only support 2D fitting')
        self.needed_parameters = set()
        self._sanity_check()

        self.data = load_data(self.data_file_name)[self.bins_on].to_numpy()
        mask = (self.data[:, 0] > config['x_clip'][0])
        mask &= (self.data[:, 0] < config['x_clip'][1])
        mask &= (self.data[:, 1] > config['y_clip'][0])
        mask &= (self.data[:, 1] < config['y_clip'][1])
        self.data = self.data[mask]

        if self.bins_type == 'meshgrid':
            # self.bins = [bin_edges_on_axis0, bin_edges_on_axis1, ...]
            #          or [num_bins_on_axis0, num_bins_on_axis1, ...]
            warning = f'The usage of meshgrid binning is highly discouraged.'
            warn(warning)
            self.component_bins_type = 'meshgrid'
            self.data_hist = make_hist_mesh_grid(
                self.data, 
                bins=jnp.asarray(self.bins), 
                weights=jnp.ones(len(self.data))
            )
        elif self.bins_type == 'equiprob':
            # self.bins = [num_bins_on_axis0, num_bins_on_axis1, ...]
            if self.dim != 2:
                raise RuntimeError('only 2D equiprob binned likelihood is supported!')
            self.bins = get_equiprob_bins_2d(self.data, 
                                             self.bins, 
                                             x_clip=config['x_clip'], 
                                             y_clip=config['y_clip'], 
                                             which_np=jnp)
            self.component_bins_type = 'irreg'
            self.data_hist = make_hist_irreg_bin_2d(
                self.data, 
                bins_x=self.bins[0], 
                bins_y=self.bins[1], 
                weights=jnp.ones(len(self.data))
            )
        else:
            raise ValueError("'bins_type' should either be meshgrid or equiprob")

    def __getitem__(self, keys):
        """Get component in likelihood"""
        return self.components[keys]

    def _sanity_check(self):
        """Check equality between number of bins group and observables"""
        if len(self.bins_on) != len(self.bins):
            raise RuntimeError('Length of bins must be the same as length of bins_on!')

    def register_component(self,
                           component_cls: Component,
                           component_name: str):
        """
        Create an appletree likelihood
        :param component_cls: class of Component
        :param component_name: name of Component
        """
        if component_name in self.components:
            raise ValueError(f'Component named {component_name} already existed!')

        # Initialize component
        component = component_cls(
            bins=self.bins,
            bins_type=self.component_bins_type
        )
        component.rate_name = component_name + '_rate'
        kwargs = dict(
            data_names=self.bins_on
        )
        if isinstance(component, ComponentSim):
            kwargs['func_name'] = component_name + '_sim'
            kwargs['data_names'] = self.bins_on + ['eff']
        component.deduce(**kwargs)
        component.compile()

        # Update components sheet
        self.components[component_name] = component

        # Update needed parameters
        self.needed_parameters |= self.components[component_name].needed_parameters

    def _simulate_model_hist(self, key, batch_size, parameters):
        """
        Histogram of simulated observables
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
            hist = hist + _hist
        return key, hist

    def get_log_likelihood(self, key, batch_size, parameters):
        """
        Get log likelihood of given parameters
        :param key: a pseudo-random number generator (PRNG) key
        :param batch_size: int of number of simulated events
        :param parameters: dict of parameters used in simulation
        """
        key, model_hist = self._simulate_model_hist(key, batch_size, parameters)
        # Poisson likelihood
        llh = float(jnp.sum(self.data_hist * jnp.log(model_hist) - model_hist))
        if np.isnan(llh):
            llh = -np.inf
        return key, llh

    def print_likelihood_summary(self, indent:str=' '*4, short=True):
        """
        Print likelihood summary: components, bins, file names
        :param indent: str of indent
        :param short: bool, whether only print short summary
        """
        print('\n'+'-'*40)

        print(f'BINNING\n')
        print(f'{indent}bins_type: {self.bins_type}')
        print(f'{indent}bins_on: {self.bins_on}')
        if not short:
            print(f'{indent}bins: {self.bins}')
        print('\n'+'-'*40)

        print(f'DATA\n')
        print(f'{indent}file_name: {self.data_file_name}')
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
                print(f'{indent*2}rate_par: {component.rate_name}')
                print(f'{indent*2}pars: {need}')
                if not short:
                    print(f'{indent*2}from_file: {component.file_name}')
            else:
                pass
            print()

        print('-'*40)
