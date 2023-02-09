import os
import copy
import json
import importlib
from datetime import datetime
import numpy as np
import emcee
import h5py

import appletree as apt
from appletree import randgen
from appletree import Parameter
from appletree import Likelihood
from appletree.utils import load_json
from appletree.share import _cached_configs, set_global_config

os.environ['OMP_NUM_THREADS'] = '1'


class Context():
    """Combine all likelihood (e.g. Rn220, Ar37),
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

        self.backend_h5 = config.get('backend_h5', None)

        self.likelihoods = {}

        self.par_config = self.get_parameter_config(config['par_config'])
        self.needed_parameters = self.update_parameter_config(config['likelihoods'])

        self.par_manager = Parameter(self.par_config)

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
        self.likelihoods[likelihood_name] = Likelihood(
            name=likelihood_name,
            **likelihood_config,
        )

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
        self.par_manager.set_parameter(parameters)

        key = randgen.get_key()
        log_posterior = 0
        for likelihood in self.likelihoods.values():
            key, log_likelihood_i = likelihood.get_log_likelihood(
                key,
                batch_size,
                self.par_manager.get_all_parameter(),
            )
            log_posterior += log_likelihood_i

        log_prior = self.par_manager.log_prior
        log_posterior += log_prior

        return log_posterior, log_prior

    def _get_backend(self, nwalkers, ndim):
        if self.backend_h5 is None:
            backend = None
            print('With no backend')
        else:
            backend = emcee.backends.HDFBackend(self.backend_h5)
            backend.reset(nwalkers, ndim)
            print(f'With h5 backend {self.backend_h5}')
        return backend

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

        backend = self._get_backend(nwalkers, ndim)
        self.sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            self.log_posterior,
            backend=backend,
            blobs_dtype=np.float32,
            parameter_names=self.par_manager.parameter_fit,
            kwargs = {'batch_size': batch_size},
        )

        result = self.sampler.run_mcmc(
            p0,
            iteration,
            store=True,
            progress=True,
        )

        self._dump_meta()
        return result

    def continue_fitting(self, context, iteration=500, batch_size=1_000_000):
        """Continue a fitting of another context

        :param context: appletree context
        :param iteration: int, number of steps to generate
        """
        # Final iteration
        final_iteration = context.sampler.get_chain()[-1, :, :]
        p0 = final_iteration.tolist()

        ndim = len(self.par_manager.parameter_fit)
        # Init sampler for current context
        backend = self._get_backend(len(p0), ndim)
        self.sampler = emcee.EnsembleSampler(
            len(p0),
            ndim,
            self.log_posterior,
            backend=backend,
            blobs_dtype=np.float32,
            parameter_names=self.par_manager.parameter_fit,
            kwargs = {'batch_size': batch_size},
        )

        result = self.sampler.run_mcmc(
            p0,
            iteration,
            store=True,
            progress=True,
            skip_initial_state_check=True,
        )

        self._dump_meta()
        return result

    def get_post_parameters(self):
        """Get parameters correspondes to max posterior"""
        logp = self.sampler.get_log_prob(flat=True)
        chain = self.sampler.get_chain(flat=True)
        mpe_parameters = chain[np.argmax(logp)]
        mpe_parameters = emcee.ensemble.ndarray_to_list_of_dicts(
            [mpe_parameters],
            self.sampler.parameter_names,
        )[0]
        parameters = copy.deepcopy(self.par_manager.get_all_parameter())
        parameters.update(mpe_parameters)
        return parameters

    def get_all_post_parameters(self, **kwargs):
        """Return all posterior parameters"""
        chain = self.sampler.get_chain(**kwargs)
        return chain

    def dump_post_parameters(self, file_name):
        """Dump max posterior parameter in .json file"""
        parameters = self.get_post_parameters()
        with open(file_name, 'w') as fp:
            json.dump(parameters, fp)

    def _dump_meta(self, metadata=None):
        """Save parameters name as attributes"""
        if metadata is None:
            metadata = {
                'version': apt.__version__,
                'date': datetime.now().strftime('%Y%m%d_%H:%M:%S'),
            }
        if self.backend_h5 is not None:
            name = self.sampler.backend.name
            with h5py.File(self.backend_h5, 'r+') as opt:
                opt[name].attrs['metadata'] = json.dumps(metadata)
                # parameters prior configuration
                opt[name].attrs['par_config'] = json.dumps(self.par_manager.par_config)
                # max posterior parameters
                opt[name].attrs['post_parameters'] = json.dumps(self.get_post_parameters())
                # the order of parameters saved in backend
                opt[name].attrs['parameter_fit'] = self.par_manager.parameter_fit
                # instructions
                opt[name].attrs['config'] = json.dumps(self.config)
                # configurations, maybe users will manually add some maps
                opt[name].attrs['_cached_configs'] = json.dumps(_cached_configs)

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
        provided = set(self.par_manager.get_all_parameter().keys())
        # We will not update unneeded parameters!
        if not provided.issubset(needed):
            mes = f'Parameter manager should provide needed parameters only, '
            mes += f'{provided - needed} not needed'
            raise RuntimeError(mes)

    def update_url_base(self, url_base):
        """Update url_base in appletree.share"""
        print(f'Updated url_base to {url_base}')
        set_global_config({'url_base': url_base})

    def get_parameter_config(self, par_config):
        """Get configuration for parameter manager

        :param par_config: str, parameters configuration file
        """
        par_config = load_json(par_config)
        return par_config

    def update_parameter_config(self, likelihoods):
        needed_parameters = set()
        needed_rate_parameters = []
        from_parameters = []
        for likelihood in likelihoods.values():
            for k, v in likelihood['copy_parameters'].items():
                # specify rate scale
                # normalization factor, for AC & ER, etc.
                self.par_config.update({k: self.par_config[v]})
                from_parameters.append(v)
                needed_parameters.add(k)
            for k in likelihood['components'].keys():
                needed_rate_parameters.append(k + '_rate')
        for p in from_parameters:
            if p not in needed_rate_parameters and p in self.par_config:
                # Drop unused parameters
                self.par_config.pop(p)
        return needed_parameters

    def set_config(self, configs):
        """Set new configuration options

        :param configs: dict, configuration file name or dictionary
        """
        if not hasattr(self, 'config'):
            self.config = dict()

        # update configuration only in this Context
        self.config.update(configs)

        # also store required configurations to appletree.share
        set_global_config(configs)

    def lineage(self, data_name: str = 'cs2'):
        """Return lineage of plugins."""
        assert isinstance(data_name, str)
        pass
