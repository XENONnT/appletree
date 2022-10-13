import os
import copy
import importlib
import numpy as np
import emcee

from appletree import randgen
from appletree import Parameter
from appletree import Likelihood
from appletree.utils import load_json
from appletree.config import get_file_path
from appletree.share import _cached_configs


class Context():
    """Combine all likelihood(e.g. Rn220, Ar37),
    handle MCMC and post-fitting analysis
    """

    def __init__(self, config):
        """Create an appletree context
        :param config: dict or str, configuration file name or dictionary
        """
        if isinstance(config, str):
            config = load_json(config)

        # url_base and configs are not mandatory
        if 'url_base' in config.keys():
            self.update_url_base(config['url_base'])

        if 'configs' in config.keys():
            self.set_config(config['configs'])

        self.likelihoods = {}

        self.par_config = self.get_parameter_config(config['par_config'])
        self.update_parameter_config(config['likelihoods'])

        self.par_manager = Parameter(self.par_config)
        self.needed_parameters = set()

        self.register_all_likelihood(config)

    def __getitem__(self, keys):
        """Get likelihood in context"""
        return self.likelihoods[keys]

    def register_all_likelihood(self, config):
        """Create all appletree likelihoods
        :param config: dict, configuration file name or dictionary
        """
        components = importlib.import_module('appletree.components')

        for key, value in config['likelihoods'].items():
            likelihood = copy.deepcopy(value)

            # update data file path
            data_file_name = likelihood["data_file_name"]
            if not os.path.exists(data_file_name):
                likelihood["data_file_name"] = data_file_name

            self.register_likelihood(key, likelihood)

            for k, v in likelihood['components'].items():
                # dynamically import components
                if isinstance(v, str):
                    self.register_component(key, getattr(components, v), k)
                else:
                    self.register_component(
                        key,
                        getattr(components, v['component_cls']),
                        k,
                        v.get('file_name', None),
                    )

    def register_likelihood(self,
                            likelihood_name,
                            likelihood_config):
        """Create an appletree likelihood
        :param likelihood_name: name of Likelihood
        :param likelihood_config: dict of likelihood configuration
        """
        if likelihood_name in self.likelihoods:
            raise ValueError(f'Likelihood named {likelihood_name} already existed!')
        self.likelihoods[likelihood_name] = Likelihood(**likelihood_config)

    def register_component(self,
                           likelihood_name,
                           component_cls,
                           component_name,
                           file_name=None):
        """Register component to likelihood
        :param likelihood_name: name of Likelihood
        :param component_cls: class of Component
        :param component_name: name of Component
        """
        self[likelihood_name].register_component(
            component_cls,
            component_name,
            file_name,
        )
        # Update needed parameters
        self.needed_parameters |= self.likelihoods[likelihood_name].needed_parameters

    def print_context_summary(self, short=True):
        """Print summary of the context."""
        self._sanity_check()

        print('\n'+'='*40)
        for key, likelihood in self.likelihoods.items():
            print(f'LIKELIHOOD {key}')
            likelihood.print_likelihood_summary(short=short)
            print('\n'+'='*40)

    def log_posterior(self, parameters, batch_size=1_000_000):
        """Get log likelihood of given parameters
        :param batch_size: int of number of simulated events
        :param parameters: dict of parameters used in simulation
        """
        self.par_manager.set_parameter_fit_from_array(parameters)

        key = randgen.get_key()
        log_posterior = 0
        for likelihood in self.likelihoods.values():
            key, log_likelihood_i = likelihood.get_log_likelihood(
                key,
                batch_size,
                self.par_manager.get_all_parameter(),
            )
            log_posterior += log_likelihood_i

        log_posterior += self.par_manager.log_prior

        return log_posterior

    def fitting(self, nwalkers=200, iteration=500, batch_size=1_000_000):
        """Fitting posterior distribution of needed parameters
        :param nwalkers: int, number of walkers in the ensemble
        :param iteration: int, number of steps to generate
        """
        self._sanity_check()

        p0 = []
        for _ in range(nwalkers):
            self.par_manager.sample_init()
            p0.append(self.par_manager.parameter_fit_array)

        ndim = len(self.par_manager.parameter_fit_array)
        self.sampler = emcee.EnsembleSampler(nwalkers,
                                             ndim,
                                             self.log_posterior,
                                             kwargs = {'batch_size': batch_size})

        result = self.sampler.run_mcmc(p0, iteration, progress=True)
        return result

    def continue_fitting(self, context, iteration=500):
        """Continue a fitting of another context
        :param context: appletree context
        :param iteration: int, number of steps to generate
        """
        # Final iteration
        final_iteration = context.sampler.get_chain()[-1, :, :]

        p0 = []
        for iwalker in final_iteration:
            self.par_manager.sample_init()

            # assign i-walker of final iteration
            context.par_manager.set_parameter_fit_from_array(iwalker)
            parameters = context.par_manager.get_all_parameter()

            self.par_manager._parameter_dict.update(parameters)
            p0.append(self.par_manager.parameter_fit_array)

        ndim = len(self.par_manager.parameter_fit_array)
        # Init sampler for current context
        self.sampler = emcee.EnsembleSampler(len(final_iteration), ndim, self.log_posterior)

        result = self.sampler.run_mcmc(p0, iteration, progress=True, skip_initial_state_check=True)
        return result

    def get_post_parameters(self):
        """Get parameters correspondes to max posterior"""
        logp = self.sampler.get_log_prob(flat=True)
        chain = self.sampler.get_chain(flat=True)
        mpe_parameters = chain[np.argmax(logp)]
        self.par_manager.set_parameter_fit_from_array(mpe_parameters)
        parameters = self.par_manager.get_all_parameter()
        return parameters

    def get_template(self,
                     likelihood_name: str,
                     component_name: str,
                     batch_size: int = 1_000_000,
                     seed: int = None):
        """Get parameters correspondes to max posterior
        :param likelihood_name: name of Likelihood
        :param component_name: name of Component
        :param batch_size: int of number of simulated events
        :param seed: random seed
        """
        parameters = self.get_post_parameters()
        key = randgen.get_key(seed=seed)

        key, result = self[likelihood_name][component_name].simulate(
            key,
            batch_size, parameters,
        )
        return result

    def _sanity_check(self):
        """Check if needed parameters are provided."""
        needed = set(self.needed_parameters)
        provided = set(self.par_manager._parameter_dict.keys())
        # We will not update unneeded parameters!
        if not needed.issubset(provided):
            mes = f'Parameter manager should provide needed parameters only, '
            mes += '{provided - needed} not needed'
            raise RuntimeError(mes)

    def update_url_base(self, url_base):
        """Update url_base in appletree.share"""
        print(f'Updated url_base to {url_base}')
        _cached_configs.updte({'url_base': url_base})

    def get_parameter_config(self, par_config):
        """Get configuration for parameter manager
        :param par_config: str, parameters configuration file
        """
        par_config = load_json(par_config)
        return par_config

    def update_parameter_config(self, likelihoods):
        for likelihood in likelihoods.values():
            for k, v in likelihood['copy_parameters'].items():
                # specify rate scale
                # normalization factor, for AC & ER, etc.
                self.par_config.update({k: self.par_config[v]})
        return self.par_config

    def set_config(self, configs):
        """Set new configuration options
        :param configs: dict, configuration file name or dictionary
        """
        if not hasattr(self, 'config'):
            self.config = dict()

        # update configuration only in this Context
        self.config.update(configs)

        # also store required configurations to appletree.share
        for k, v in configs.items():
            file_path = get_file_path(v)
            _cached_configs.update({k: file_path})

    def lineage(self, data_name: str = 'cs2'):
        """Return lineage of plugins."""
        assert isinstance(data_name, str)
        pass
