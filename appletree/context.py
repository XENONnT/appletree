import os

import emcee

from appletree import randgen
from appletree import Parameter
from appletree import Likelihood
from appletree.utils import load_json
from appletree.share import DATAPATH, PARPATH
from appletree.components import *


class Context():
    def __init__(self, parameter_config):
        self.likelihoods = {}
        self.par_manager = Parameter(parameter_config)
        self.needed_parameters = set()

    def __getitem__(self, keys):
        return self.likelihoods[keys]

    def register_likelihood(self, 
                            likelihood_name, 
                            likelihood_config):
        self.likelihoods[likelihood_name] = Likelihood(**likelihood_config)

    def register_component(self, 
                           likelihood_name, 
                           component_cls, 
                           component_name, 
                           rate_name=None):
        self[likelihood_name].register_component(
            component_cls, 
            component_name, 
            rate_name
        )
        self.needed_parameters |= self.likelihoods[likelihood_name].needed_parameters

    def print_context_summary(self, short=True):
        self.sanity_check()
        print('\n'+'='*80)
        for key, likelihood in self.likelihoods.items():
            print(f'LIKELIHOOD {key}')
            likelihood.print_likelihood_summary(short=short)
            print('\n'+'='*80)

    def sanity_check(self):
        needed = set(self.needed_parameters)
        provided = set(self.par_manager._parameter_dict.keys())
        # We will not update unneeded parameters!
        assert needed == provided, f'Parameter manager should provide needed parameters only, {provided - needed} not needed'

    def log_posterior(self, pars, batch_size=int(1e6)):
        self.par_manager.set_parameter_fit_from_array(pars)

        key = randgen.get_key()
        log_posterior = 0
        for likelihood in self.likelihoods.values():
            key, log_likelihood_i = likelihood.get_log_likelihood(key, batch_size, self.par_manager.get_all_parameter())
            log_posterior += log_likelihood_i

        log_posterior += self.par_manager.log_prior

        return log_posterior

    def fitting(self, nwalkers=200, iteration=500):
        self.sanity_check()

        p0 = []
        for _ in range(nwalkers):
            self.par_manager.sample_init()
            p0.append(self.par_manager.parameter_fit_array)

        ndim = len(self.par_manager.parameter_fit_array)
        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior)

        result = self.sampler.run_mcmc(p0, iteration, progress=True)
        return result

    def continue_fitting(self, context, iteration=500):
        final_iteration = context.sampler.get_chain()[-1, :, :]

        p0 = []
        for iwalker in final_iteration:
            self.par_manager.sample_init()

            context.par_manager.set_parameter_fit_from_array(iwalker)
            parameters = context.par_manager.get_all_parameter()

            self.par_manager._parameter_dict.update(parameters)
            p0.append(self.par_manager.parameter_fit_array)

        ndim = len(self.par_manager.parameter_fit_array)
        self.sampler = emcee.EnsembleSampler(len(final_iteration), ndim, self.log_posterior)

        result = self.sampler.run_mcmc(p0, iteration, progress=True, skip_initial_state_check=True)
        return result

    def get_post_parameters(self, burn_in=None):
        # Default burn-in is half of the iteration, feel free to change it
        if burn_in is None:
            burn_in = int(self.sampler.iteration * 0.5)
        self.par_manager.set_parameter_fit_from_array(self.sampler.get_chain()[burn_in:].mean(axis=(0, 1)))
        parameters = self.par_manager.get_all_parameter()
        return parameters

    def get_template(self, 
                     likelihood_name, 
                     component_name, 
                     burn_in=None, 
                     batch_size=int(1e6), 
                     seed=None):
        parameters = self.get_post_parameters(burn_in)
        key = randgen.get_key(seed=seed)

        key, result = self[likelihood_name][component_name].simulate(
            key, 
            batch_size, parameters
        )
        return result


class ContextRn220(Context):
    """
    A specified context for ER response by Rn220 fit
    """
    def __init__(self):
        par_config = load_json(os.path.join(PARPATH, 'apt_sr0_er.json'))
        # specify rate scale
        par_config.update({'rn220_ac_rate': par_config['ac_rate']})
        par_config.update({'rn220_er_rate': par_config['er_rate']})
        par_config.pop('ac_rate')
        par_config.pop('er_rate')
        super().__init__(par_config)

        rn_config = dict(
            data_file_name = os.path.join(
                DATAPATH, 
                'data_XENONnT_Rn220_v8_strax_v1.2.2_straxen_v1.7.1_cutax_v1.9.0.csv'
            ), 
            bins_type = 'equiprob',
            bins_on = ['cs1', 'cs2'],
            bins = [15, 15],
            x_clip = [0, 100],
            y_clip = [2e2, 1e4],
        )
        self.register_likelihood('rn220_llh', rn_config)
        self.register_component('rn220_llh', ERBand, 'rn220_er', 'rn220_er_rate')
        self.register_component('rn220_llh', AC, 'rn220_ac', 'rn220_ac_rate')


class ContextER(Context):
    """
    A specified context for ER response by Rn220 & Ar37 combined fit
    """
    def __init__(self):
        par_config = load_json(os.path.join(PARPATH, 'apt_sr0_er.json'))
        # specify rate scale
        par_config.update({'rn220_ac_rate': par_config['ac_rate']})
        par_config.update({'rn220_er_rate': par_config['er_rate']})
        par_config.update({'ar37_er_rate': par_config['er_rate']})
        par_config.pop('ac_rate')
        par_config.pop('er_rate')
        super().__init__(par_config)

        rn_config = dict(
            data_file_name = os.path.join(
                DATAPATH, 
                'data_XENONnT_Rn220_v8_strax_v1.2.2_straxen_v1.7.1_cutax_v1.9.0.csv'
            ), 
            bins_type = 'equiprob',
            bins_on = ['cs1', 'cs2'],
            bins = [15, 15],
            x_clip = [0, 100],
            y_clip = [2e2, 1e4],
        )
        self.register_likelihood('rn220_llh', rn_config)
        self.register_component('rn220_llh', ERBand, 'rn220_er', 'rn220_er_rate')
        self.register_component('rn220_llh', AC, 'rn220_ac', 'rn220_ac_rate')

        ar_config = dict(
            data_file_name = os.path.join(
                DATAPATH, 
                'data_XENONnT_Ar37_v2_1e4_events_2sig_strax_v1.2.2_straxen_v1.7.1_cutax_v1.9.0.csv'
            ), 
            bins_type = 'equiprob',
            bins_on = ['cs1', 'cs2'],
            bins = [20, 20],
            x_clip = [0, 50],
            y_clip = [1250, 2200],
        )
        self.register_likelihood('ar37_llh', ar_config)
        self.register_component('ar37_llh', ERPeak, 'ar37_er', 'ar37_er_rate')
