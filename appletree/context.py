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

    def register_likelihood(self, 
                            likelihood_name, 
                            likelihood_config):
        self.likelihoods[likelihood_name] = Likelihood(**likelihood_config)

    def register_component(self, 
                           likelihood_name, 
                           component_cls, 
                           component_name, 
                           rate_name=None):
        self.likelihoods[likelihood_name].register_component(
            component_cls, 
            component_name, 
            rate_name
        )

    def print_context_summary(self, short=True):
        print('\n'+'='*80)
        for key, likelihood in self.likelihoods.items():
            print(f'LIKELIHOOD {key}')
            likelihood.print_likelihood_summary(short=short)
            print('\n'+'='*80)

    def log_posterior(self, pars, batch_size=int(1e6)):
        # TODO: shall we update nuisance parameters?
        self.par_manager.set_parameter_fit_from_array(pars)

        key = randgen.get_key()
        log_posterior = 0
        for likelihood in self.likelihoods.values():
            key, log_likelihood_i = likelihood.get_log_likelihood(key, batch_size, self.par_manager.get_all_parameter())
        log_posterior += log_likelihood_i

        log_prior = self.par_manager.log_prior
        log_posterior += log_prior

        return log_posterior

    def fitting(self, nwalkers=200, steps=500):
        ndim = len(self.par_manager.parameter_fit_array)
        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior)

        p0 = []
        for _ in range(nwalkers):
            self.par_manager.sample_init()
            p0.append(self.par_manager.parameter_fit_array)

        _ = self.sampler.run_mcmc(p0, steps, progress=True)

    def get_post_parameters(self):
        # TODO: how many iteration is burning
        self.par_manager.set_parameter_fit_from_array(self.sampler.get_chain()[-1].mean(axis=0))
        parameters = self.par_manager.get_all_parameter()
        return parameters

    def get_template(self, 
                     likelihood_name, 
                     component_name, 
                     batch_size=int(1e6), 
                     seed=None):
        parameters = self.get_post_parameters()
        key = randgen.get_key(seed=seed)

        key, result = self.likelihoods[likelihood_name].components[component_name].simulate(
            key, 
            batch_size, parameters)
        return result


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

        self.register_component('rn220_llh', AC, 'rn220_ac', 'rn220_ac_rate')
        self.register_component('rn220_llh', ERBand, 'rn220_er', 'rn220_er_rate')
        self.register_component('ar37_llh', ERPeak, 'ar37_er', 'ar37_er_rate')
