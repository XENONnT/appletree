import copy
import json

import numpy as np

class Parameter():
    _parameter_fixed = []
    _parameter_fit = []

    def __init__(self, parameter_config):
        if isinstance(parameter_config, str):
            with open(parameter_config, 'r') as file:
                self.par_config = json.load(file)
        elif isinstance(parameter_config, dict):
            self.par_config = copy.deepcopy(parameter_config)
        else:
            raise RuntimeError('Parameter configuration should be file name or dictionary')

        self.init_parameter()

    def init_parameter(self, seed=None):
        self._parameter_dict = {par_name : 0 for par_name in self.par_config.keys()}

        for par_name in self.par_config:
            if self.par_config[par_name]['prior_type'] == 'fixed':
                self._parameter_fixed.append(par_name)
            else:
                self._parameter_fit.append(par_name)

        if seed is not None:
            np.random.seed(seed)
        self.sample_prior()

    def sample_prior(self):
        for par_name in self._parameter_dict:
            try:
                setting = self.par_config[par_name]
            except:
                raise RuntimeError(f'Requested parameter "{par_name}" not in given configuration')

            if setting['prior_type'] == 'norm':
                kwargs = {
                    'loc' : setting['prior_args']['mean'],
                    'scale' : setting['prior_args']['std'],
                }
                val = np.random.normal(**kwargs)
                self._parameter_dict[par_name] = np.clip(val, *setting['allowed_range'])
            elif setting['prior_type'] == 'uniform':
                kwargs = {
                    'low' : setting['prior_args']['lower'],
                    'high' : setting['prior_args']['upper'],
                }
                val = np.random.uniform(**kwargs)
                self._parameter_dict[par_name] = np.clip(val, *setting['allowed_range'])
            elif setting['prior_type'] == 'free': # we sample from init guessing if it's free-prior
                kwargs = {
                    'loc' : setting['init_mean'],
                    'scale' : setting['init_std'],
                }
                val = np.random.normal(**kwargs)
                self._parameter_dict[par_name] = np.clip(val, *setting['allowed_range'])
            elif setting['prior_type'] == 'fixed':
                self._parameter_dict[par_name] = setting['prior_args']['val']
                
    def sample_init(self):
        for par_name in self._parameter_dict:
            setting = self.par_config[par_name]
            
            if setting['prior_type'] == 'fixed':
                self._parameter_dict[par_name] = setting['prior_args']['val']
            else:
                kwargs = {
                    'loc' : setting['init_mean'],
                    'scale' : setting['init_std'],
                }
                val = np.random.normal(**kwargs)
                self._parameter_dict[par_name] = np.clip(val, *setting['allowed_range'])

    @property
    def log_prior(self):
        log_prior = 0

        for par_name in self._parameter_fit:
            val = self._parameter_dict[par_name]
            setting = self.par_config[par_name]

            if val < setting['allowed_range'][0] or val > setting['allowed_range'][1]:
                log_prior += -1e30
            elif setting['prior_type'] == 'norm':
                mean = setting['prior_args']['mean']
                std = setting['prior_args']['std']
                log_prior += - (val - mean)**2 / 2 / std**2
            elif setting['prior_type'] == 'free':
                pass
            elif setting['prior_type'] == 'uniform':
                pass

        return log_prior

    def check_parameter_exist(self, keys, return_not_exist=False):
        """
        Check whether the keys exist in parameters.
        
        keys             : Parameter names. Can be a single str, or a list of str.
        return_not_exist : If False, function will return a bool if all keys exist. If True, function will additionally return the not existing list of keys.
        """
        if isinstance(keys, list):
            not_exist = []
            for key in keys:
                if not key in self._parameter_dict:
                    not_exist.append(key)
            all_exist = (not_exist==[])
            if return_not_exist:
                return (all_exist, not_exist)
            else:
                return (all_exist)
        elif isinstance(keys, str):
            if return_not_exist:
                return (keys in self._parameter_dict, keys)
            else:
                return (keys in self._parameter_dict)
        elif isinstance(keys, dict):
            return self.check_parameter_exist(list(keys.keys()), return_not_exist)

        else:
            raise ValueError("keys must be a str or a list of str!")

    def set_parameter(self, keys, vals=None):
        """
        Set parameter values.
        
        keys : Parameter names. Can be a single str, or a list of str, or a dict.
               If str, vals must be int or float.
               If list, vals must have the same length.
               If dict, vals will be overwritten as keys.values() and ignore vals.
        vals : values to be set.       
        """
        all_exist, not_exist = self.check_parameter_exist(keys, return_not_exist=True)
        assert all_exist, f"{not_exist} not found!"

        if isinstance(keys, list):
            assert len(keys) == len(vals), "keys must have the same length as vals!"
            for key, val in zip(keys, vals):
                self._parameter_dict[key] = val
        elif isinstance(keys, dict):
            self.set_parameter(list(keys.keys()), keys.values())
        elif isinstance(keys, str):
            assert isinstance(vals, (float, int)), "if there is only one key, val must be either float or int!"
            self._parameter_dict[keys] = vals
        else:
            raise ValueError("keys must be a str or a list of str!")

    def set_parameter_fit_from_array(self, arr):
        assert len(arr) == len(self._parameter_fit), f"the length of arr must be the same as length of parameter to fit {len(self._parameter_fit)}!"

        update = {par_name : val for par_name, val in zip(self._parameter_fit, arr)}
        self.set_parameter(update)

    def get_parameter(self, keys):
        """
        Return parameter values.
        
        keys : Parameter names. Can be a single str, or a list of str.
        """
        all_exist, not_exist = self.check_parameter_exist(keys, return_not_exist=True)
        assert all_exist, f"{not_exist} not found!"

        return self.__getitem__(keys)

    def __getitem__(self, keys):
        if isinstance(keys, list):
            return np.array([self._parameter_dict[key] for key in keys])
        elif isinstance(keys, str):
            return self._parameter_dict[keys]
        else:
            raise ValueError("keys must be a str or a list of str!")

    @property
    def parameter_fit_array(self):
        return self.get_parameter(self._parameter_fit)

    def get_all_parameter(self):
        """
        Return all parameters as a dict.     
        """
        return self._parameter_dict
