import copy
import json

import numpy as np

from appletree.randgen import TwoHalfNorm


class Parameter:
    """Parameter handler to update parameters and calculate prior."""

    def __init__(self, parameter_config, multi=False):
        """Initialization.

        :param parameter_config: can be either

          * str: the json file name where the config is stored.
          * dict: config dictionary if multi=False. If multi=True, dictionary
                  for each likelihood name, where the value is the json file name
                  or the config dictionary.
        
        :param multi: if True, use different parameters for each likelihood.

        """
        if isinstance(parameter_config, str):
            with open(parameter_config, "r") as file:
                self.par_config = json.load(file)
        elif isinstance(parameter_config, dict):
            if not multi:
                self.par_config = copy.deepcopy(parameter_config)
            else:
                self.read_multiple_parameters(parameter_config)
        else:
            raise RuntimeError("Parameter configuration should be file name or dictionary")

        self._parameter_fixed = set()
        self._parameter_fit = set()
        self.init_parameter()
    
    def read_multiple_parameters(self, parameter_config):
        """Handle multiple parameter input."""
        self.multi_par_config = {}
        for llh_name in parameter_config.keys():
            if isinstance(parameter_config[llh_name], str):
                with open(parameter_config[llh_name], "r") as file:
                    self.multi_par_config[llh_name] = json.load(file)
            elif isinstance(parameter_config[llh_name], dict):
                self.multi_par_config[llh_name] = copy.deepcopy(
                    parameter_config[llh_name]
                )
            else:
                raise RuntimeError("Parameter configuration should be file name or dictionary")
        self.merge_multiple_parameters()
    
    def merge_multiple_parameters(self):
        """Merge parameters according to the "shared" attribute."""
        self.par_config = {}
        shared_par_names = set()
        for llh_name, single_par_config in self.multi_par_config.items():
            for par_name, par_setting in single_par_config.items():
                if par_name in self.par_config:
                    if par_setting.get("shared", False):
                        if par_setting != self.par_config[par_name]:
                            raise ValueError(
                                f"""
                                Inconsistent parameter settings for {par_name}! 
                                If this is intentional, set shared=False.
                                """
                            )
                    else:
                        self.par_config[f"{par_name}_{self.par_config[par_name]["llh_name"]}"] = (
                            self.par_config[par_name]
                        )
                        self.par_config[f"{par_name}_{llh_name}"] = par_setting
                        shared_par_names.update(par_name)
                else:
                    self.par_config[par_name] = par_setting
                    if not par_setting.get("shared", False):
                        self.par_config[par_name]["llh_name"] = llh_name
        for shared_par_name in shared_par_names:
            del self.par_config[shared_par_name]                

    def init_parameter(self, seed=None):
        """Initializing parameters by sampling prior. If the prior is free, then sampling from the
        initial guess.

        :param seed: integer, sent to np.random.seed(seed)

        """
        self._parameter_dict = {par_name: 0 for par_name in self.par_config.keys()}

        for par_name in self.par_config.keys():
            if self.par_config[par_name]["prior_type"] == "fixed":
                self._parameter_fixed.add(par_name)
            else:
                self._parameter_fit.add(par_name)
        # Parameters name set is not sorted
        self._parameter_fixed = set(self._parameter_fixed)
        self._parameter_fit = set(self._parameter_fit)

        if seed is not None:
            np.random.seed(seed)
        self.sample_prior()

    @property
    def parameter_fit(self):
        """Return sorted list of parameters name waiting for fitting."""
        return sorted(self._parameter_fit)

    def sample_prior(self):
        """Sampling parameters from prior and set self._parameter_dict.

        If the prior is free, then sampling from the initial guess.

        """
        for par_name in self._parameter_dict:
            try:
                setting = self.par_config[par_name]
            except KeyError:
                raise RuntimeError(f'Requested parameter "{par_name}" not in given configuration')

            args = setting["prior_args"]
            prior_type = setting["prior_type"]

            if prior_type == "norm":
                kwargs = {
                    "loc": args["mean"],
                    "scale": args["std"],
                }
                val = np.random.normal(**kwargs)
                self._parameter_dict[par_name] = np.clip(val, *setting["allowed_range"])
            elif prior_type == "twohalfnorm":
                kwargs = {
                    "mu": args["mu"],
                    "sigma_pos": args["sigma_pos"],
                    "sigma_neg": args["sigma_neg"],
                }
                val = TwoHalfNorm.rvs(**kwargs)
                self._parameter_dict[par_name] = np.clip(val, *setting["allowed_range"])
            elif prior_type == "uniform":
                kwargs = {
                    "low": args["lower"],
                    "high": args["upper"],
                }
                val = np.random.uniform(**kwargs)
                self._parameter_dict[par_name] = np.clip(val, *setting["allowed_range"])
            elif prior_type == "free":
                kwargs = {
                    "loc": setting["init_mean"],
                    "scale": setting["init_std"],
                }
                val = np.random.normal(**kwargs)
                self._parameter_dict[par_name] = np.clip(val, *setting["allowed_range"])
            elif prior_type == "fixed":
                self._parameter_dict[par_name] = args["val"]

    def sample_init(self):
        """Samping parameters from initial guess clipped by the allowed_range and set
        self._parameter_dict."""
        for par_name in self._parameter_dict:
            try:
                setting = self.par_config[par_name]
            except KeyError:
                raise RuntimeError(f'Requested parameter "{par_name}" not in given configuration')

            args = setting["prior_args"]
            prior_type = setting["prior_type"]

            if prior_type == "fixed":
                self._parameter_dict[par_name] = args["val"]
            else:
                kwargs = {
                    "loc": setting["init_mean"],
                    "scale": setting["init_std"],
                }
                val = np.random.normal(**kwargs)
                self._parameter_dict[par_name] = np.clip(val, *setting["allowed_range"])

    @property
    def log_prior(self):
        """Return log prior.

        If any parameter is out of allowed_range return -np.inf.

        """
        log_prior = 0

        for par_name in self._parameter_fit:
            val = self._parameter_dict[par_name]
            setting = self.par_config[par_name]

            args = setting["prior_args"]
            prior_type = setting["prior_type"]

            if val < setting["allowed_range"][0] or val > setting["allowed_range"][1]:
                log_prior += -np.inf
            elif prior_type == "norm":
                mean = args["mean"]
                std = args["std"]
                log_prior += -((val - mean) ** 2) / 2 / std**2
            elif prior_type == "twohalfnorm":
                mu = args["mu"]
                sigma_pos = args["sigma_pos"]
                sigma_neg = args["sigma_neg"]
                log_prior += TwoHalfNorm.logpdf(
                    x=val,
                    mu=mu,
                    sigma_pos=sigma_pos,
                    sigma_neg=sigma_neg,
                )
            elif prior_type == "free":
                pass
            elif prior_type == "uniform":
                pass

        return log_prior

    def check_parameter_exist(self, keys, return_not_exist=False):
        """Check whether the keys exist in parameters.

        :param keys: Parameter names. Can be a single str, or a list of str. :param
        return_not_exist: If False, function will return a bool if all keys exist.     If True,
        function will additionally return the list of the not existing keys.

        """
        if isinstance(keys, (set, list)):
            not_exist = []
            for key in keys:
                if key not in self._parameter_dict:
                    not_exist.append(key)
            all_exist = not_exist == []
            if return_not_exist:
                return (all_exist, not_exist)
            else:
                return all_exist
        elif isinstance(keys, str):
            if return_not_exist:
                return (keys in self._parameter_dict, keys)
            else:
                return keys in self._parameter_dict
        elif isinstance(keys, dict):
            return self.check_parameter_exist(list(keys.keys()), return_not_exist)

        else:
            raise ValueError("keys must be a str or a list of str!")

    def set_parameter(self, keys, vals=None):
        """Set parameter values.

        :param keys: Parameter names. Can be either

          * str: vals must be int or float.
          * list: vals must have the same length.
          * dict: vals will be overwritten as keys.values().
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

    def get_parameter(self, keys):
        """Return parameter values.

        :param keys: Parameter names. Can be a single str, or a list of str.

        """
        all_exist, not_exist = self.check_parameter_exist(keys, return_not_exist=True)
        assert all_exist, f"{not_exist} not found!"

        return self.__getitem__(keys)

    def __getitem__(self, keys):
        """__getitem__, keys can be str/list/set."""
        if isinstance(keys, (set, list)):
            return np.array([self._parameter_dict[key] for key in keys])
        elif isinstance(keys, str):
            return self._parameter_dict[keys]
        else:
            raise ValueError("keys must be a str or a list of str!")

    @property
    def parameter_fit_array(self):
        """Return non-fixed parameters, ordered by self._parameter_fit."""
        return self.get_parameter(self.parameter_fit)

    def get_all_parameter(self):
        """Return all parameters as a dict."""
        return self._parameter_dict

    def get_parameter_for_likelihood(self, llh_name):
        """Return all parameters for a certain likelihood, given multi=True."""
        assert hasattr(self, "multi_par_config"), "Cannot call get_parameter_for_likelihood for single parameter."
        if llh_name not in self.multi_par_config:
            raise ValueError(f"Likelihood {llh_name} does not have a parameter configuration!")
        parameter_for_likelihood = {}
        for par_name in self.multi_par_config[llh_name].keys():
            if par_name in self._parameter_dict:
                parameter_for_likelihood[par_name] = self._parameter_dict[par_name]
            elif f"{par_name}_{llh_name}" in self._parameter_dict:
                parameter_for_likelihood[par_name] = self._parameter_dict[f"{par_name}_{llh_name}"]
            else:
                raise ValueError(f"{par_name} not found for likelihood {llh_name}!")
        return parameter_for_likelihood