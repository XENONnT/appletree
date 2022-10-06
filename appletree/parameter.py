import copy
import json

import numpy as np


class Parameter():
    """Parameter handler to update parameters and calculate prior."""

    def __init__(self, parameter_config):
        """Initialization
        :param parameter_config: can be either:
        - str: the json file name where the config is stored.
        - dict: config dictionary.
        """
        if isinstance(parameter_config, str):
            with open(parameter_config, 'r') as file:
                self.par_config = json.load(file)
        elif isinstance(parameter_config, dict):
            self.par_config = copy.deepcopy(parameter_config)
        else:
            raise RuntimeError('Parameter configuration should be file name or dictionary')

        self._parameter_fixed = set()
        self._parameter_fit = set()
        self.init_parameter()

    def init_parameter(self, seed=None):
        """Initializing parameters by sampling prior.
        If the prior is free, then sampling from the initial guess.
        :param seed: integer, sent to np.random.seed(seed)
        """
        self._parameter_dict = {par_name: 0 for par_name in self.par_config.keys()}

        for par_name in self.par_config.keys():
            if self.par_config[par_name]['prior_type'] == 'fixed':
                self._parameter_fixed.add(par_name)
            else:
                self._parameter_fit.add(par_name)

        if seed is not None:
            np.random.seed(seed)
        self.sample_prior()

    def sample_prior(self):
        """Sampling parameters from prior and set self._parameter_dict.
        If the prior is free, then sampling from the initial guess.
        """
        for par_name in self._parameter_dict:
            try:
                setting = self.par_config[par_name]
            except KeyError:
                raise RuntimeError(f'Requested parameter "{par_name}" not in given configuration')

            args = setting['prior_args']
            prior_type = setting['prior_type']

            if prior_type == 'norm':
                kwargs = {
                    'loc': args['mean'],
                    'scale': args['std'],
                }
                val = np.random.normal(**kwargs)
                self._parameter_dict[par_name] = np.clip(val, *setting['allowed_range'])
            elif prior_type == 'uniform':
                kwargs = {
                    'low': args['lower'],
                    'high': args['upper'],
                }
                val = np.random.uniform(**kwargs)
                self._parameter_dict[par_name] = np.clip(val, *setting['allowed_range'])
            elif prior_type == 'free':
                kwargs = {
                    'loc': setting['init_mean'],
                    'scale': setting['init_std'],
                }
                val = np.random.normal(**kwargs)
                self._parameter_dict[par_name] = np.clip(val, *setting['allowed_range'])
            elif prior_type == 'fixed':
                self._parameter_dict[par_name] = args['val']

    def sample_init(self):
        """Samping parameters from initial guess clipped
        by the allowed_range and set self._parameter_dict.
        """
        for par_name in self._parameter_dict:
            try:
                setting = self.par_config[par_name]
            except KeyError:
                raise RuntimeError(f'Requested parameter "{par_name}" not in given configuration')

            args = setting['prior_args']
            prior_type = setting['prior_type']

            if prior_type == 'fixed':
                self._parameter_dict[par_name] = args['val']
            else:
                kwargs = {
                    'loc': setting['init_mean'],
                    'scale': setting['init_std'],
                }
                val = np.random.normal(**kwargs)
                self._parameter_dict[par_name] = np.clip(val, *setting['allowed_range'])

    @property
    def log_prior(self):
        """Return log prior. If any parameter is out of allowed_range return -np.inf."""
        log_prior = 0

        for par_name in self._parameter_fit:
            val = self._parameter_dict[par_name]
            setting = self.par_config[par_name]

            args = setting['prior_args']
            prior_type = setting['prior_type']

            if val < setting['allowed_range'][0] or val > setting['allowed_range'][1]:
                log_prior += -np.inf
            elif prior_type == 'norm':
                mean = args['mean']
                std = args['std']
                log_prior += - (val - mean)**2 / 2 / std**2
            elif prior_type == 'free':
                pass
            elif prior_type == 'uniform':
                pass

        return log_prior

    def check_parameter_exist(self, keys, return_not_exist=False):
        """Check whether the keys exist in parameters.
        :param keys: Parameter names. Can be a single str, or a list of str.
        :param return_not_exist: If False, function will return a bool if all keys exist.
        If True, function will additionally return the list of the not existing keys.
        """
        if isinstance(keys, (set, list)):
            not_exist = []
            for key in keys:
                if key not in self._parameter_dict:
                    not_exist.append(key)
            all_exist = (not_exist == [])
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
        """Set parameter values.
        :param keys: Parameter names. Can be either
        - str: vals must be int or float.
        - list: vals must have the same length.
        - dict: vals will be overwritten as keys.values().
        :param vals: Values to be set.
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
            assert isinstance(vals, (float, int)), "val must be either float or int!"
            self._parameter_dict[keys] = vals
        else:
            raise ValueError("keys must be a str or a list of str!")

    def set_parameter_fit_from_array(self, arr):
        """Set non-fixed parameters by an array. The order is given by self._parameter_fit."""
        if len(arr) != len(self._parameter_fit):
            mes = f"the length of arr must be the same as length of parameter to fit {len(self._parameter_fit)}!"
            raise ValueError(mes)

        update = {par_name: val for par_name, val in zip(self._parameter_fit, arr)}
        self.set_parameter(update)

    def get_parameter(self, keys):
        """Return parameter values.
        :param keys: Parameter names. Can be a single str, or a list of str.
        """
        all_exist, not_exist = self.check_parameter_exist(keys, return_not_exist=True)
        assert all_exist, f"{not_exist} not found!"

        return self.__getitem__(keys)

    def __getitem__(self, keys):
        """__getitem__, keys can be str/list/set"""
        if isinstance(keys, (set, list)):
            return np.array([self._parameter_dict[key] for key in keys])
        elif isinstance(keys, str):
            return self._parameter_dict[keys]
        else:
            raise ValueError("keys must be a str or a list of str!")

    @property
    def parameter_fit_array(self):
        """Return non-fixed parameters, ordered by self._parameter_fit."""
        return self.get_parameter(self._parameter_fit)

    def get_all_parameter(self):
        """Return all parameters as a dict."""
        return self._parameter_dict
