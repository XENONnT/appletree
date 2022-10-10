import os
import copy
import json
import numpy as np

import emcee

from appletree import randgen
from appletree import Parameter
from appletree import Likelihood
from appletree.utils import load_json
from appletree.share import DATAPATH, PARPATH, CONFPATH
from appletree.components import ERBand, ERPeak, AC


class Context():
    """Combine all likelihood(e.g. Rn220, Ar37),
    handle MCMC and post-fitting analysis
    """

    def __init__(self, config):
        """Create an appletree context
        :param config: dict or str, configuration file name or dictionary
        """
        self.likelihoods = {}
        if isinstance(config, str):
            with open(config, 'r') as file:
                config = json.load(file)
        parameter_config = self.get_parameter_config(config)
        self.par_manager = Parameter(parameter_config)
        self.needed_parameters = set()

        self.register_all_likelihood(config)

    def __getitem__(self, keys):
        """Get likelihood in context"""
        return self.likelihoods[keys]

    def register_all_likelihood(self, config):
        """Create all appletree likelihoods
        :param config: dict, configuration file name or dictionary
        """
        for key, value in config['likelihood'].items():
            likelihood = copy.deepcopy(value)

            # update data file path
            data_file_name = likelihood["data_file_name"]
            if not os.path.exists(data_file_name):
                likelihood["data_file_name"] = os.path.join(
                    DATAPATH,
                    data_file_name,
                )

            self.register_likelihood(key, likelihood)

            for k, v in likelihood['components'].items():
                self.register_component(key, globals()[k], v)

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
                           component_name):
        """Register component to likelihood
        :param likelihood_name: name of Likelihood
        :param component_cls: class of Component
        :param component_name: name of Component
        """
        self[likelihood_name].register_component(
            component_cls,
            component_name,
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

    def get_parameter_config(self, config):
        """Get configuration for parameter manager
        :param config: dict, configuration file name or dictionary
        """
        if os.path.exists(config['par_config']):
            par_config = load_json(config['par_config'])
        else:
            par_config = load_json(os.path.join(PARPATH, 'apt_er_sr0.json'))

        for likelihood in config['likelihood'].values():
            for k, v in likelihood['copy_parameters'].items():
                # specify rate scale
                # normalization factor, for AC & ER, etc.
                par_config.update({k: par_config[v]})
        return par_config

    def set_config(self, config):
        """Set new configuration options
        :param config: dict, configuration file name or dictionary
        """
        if not hasattr(self, 'config'):
            self.config = dict()

        self.config.update(config)

    def get_single_plugin(self, run_id, data_name):
        """
        Return a single fully initialized plugin that produces
        data_name for run_id. For use in custom processing.
        """
        plugin = self._get_plugins((data_name,), run_id)[data_name]
        self._set_plugin_config(plugin, run_id, tolerant=False)
        plugin.setup()
        return plugin

    def _set_plugin_config(self, plugin):
        pass

    def show_config(self):
        pass

    def lineage(self):
        pass


class ContextRn220(Context):
    """A specified context for ER response by Rn220 fit"""

    def __init__(self):
        """Initialization"""
        config = os.path.join(CONFPATH, 'apt_config_rn220_sr0.json')
        super().__init__(config)


class ContextER(Context):
    """A specified context for ER response by Rn220 & Ar37 combined fit"""

    def __init__(self):
        """Initialization"""
        config = os.path.join(CONFPATH, 'apt_config_rn220_ar37_sr0.json')
        super().__init__(config)
