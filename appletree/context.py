from warnings import warn
import os
import copy
import json
import importlib
from datetime import datetime
from typing import Set, Optional

import numpy as np
import h5py
import emcee
from strax import deterministic_hash

import appletree as apt
from appletree import randgen
from appletree import Parameter
from appletree.utils import JSON_OPTIONS, load_json, get_file_path
from appletree.share import _cached_configs, set_global_config

os.environ["OMP_NUM_THREADS"] = "1"


class Context:
    """Combine all likelihood (e.g. Rn220, Ar37), handle MCMC and post-fitting analysis."""

    def __init__(self, instruct, par_config=None):
        """Create an appletree context.

        Args:
            instruct: dict or str, instruct file name or dictionary.

        """
        if isinstance(instruct, str):
            instruct = load_json(instruct)

        # url_base and configs are not mandatory
        if "url_base" in instruct.keys():
            self.update_url_base(instruct["url_base"])

        self.instruct = instruct
        self.config = instruct.get("configs", {})
        set_global_config(self.config)

        self.backend_h5 = instruct.get("backend_h5", None)

        self.likelihoods = dict()

        if par_config is not None:
            self.par_config = copy.deepcopy(par_config)
            print("Manually set a parameters list!")
        else:
            self.par_config = self.get_parameter_config(instruct["par_config"])
        self.needed_parameters = self.update_parameter_config(instruct["likelihoods"])

        self.par_manager = Parameter(self.par_config)

        self.register_all_likelihood(instruct)

    @classmethod
    def from_backend(cls, backend_h5_file_name):
        """Initialize context from a backend_h5 file."""
        with h5py.File(get_file_path(backend_h5_file_name)) as file:
            instruct = eval(file["mcmc"].attrs["instruct"])
            nwalkers = file["mcmc"].attrs["nwalkers"]
            batch_size = file["mcmc"].attrs["batch_size"]
        tree = cls(instruct)
        # Overwrite the backend_h5 file name. If the user renames the
        # backend file then it could be different from the one in intruct
        tree.backend_h5 = backend_h5_file_name
        tree.pre_fitting(nwalkers, batch_size=batch_size)
        return tree

    def __getitem__(self, keys):
        """Get likelihood in context."""
        return self.likelihoods[keys]

    def register_all_likelihood(self, config):
        """Create all appletree likelihoods.

        Args:
            config: dict, configuration file name or dictionary.

        """
        components = importlib.import_module("appletree.components")

        for key, value in config["likelihoods"].items():
            likelihood = copy.deepcopy(value)

            self.register_likelihood(key, likelihood)

            for k, v in likelihood["components"].items():
                # dynamically import components
                if isinstance(v, str):
                    self.register_component(key, getattr(components, v), k)
                else:
                    self.register_component(
                        key,
                        getattr(components, v["component_cls"]),
                        k,
                        v.get("file_name", None),
                    )

    def register_likelihood(self, likelihood_name, likelihood_config):
        """Create an appletree likelihood.

        Args:
            likelihood_name: name of Likelihood.
            likelihood_config: dict of likelihood configuration.

        """
        if likelihood_name in self.likelihoods:
            raise ValueError(f"Likelihood named {likelihood_name} already existed!")
        likelihood = getattr(apt, likelihood_config.get("type", "Likelihood"))
        self.likelihoods[likelihood_name] = likelihood(
            name=likelihood_name,
            **likelihood_config,
        )

    def register_component(self, likelihood_name, component_cls, component_name, file_name=None):
        """Register component to likelihood.

        Args:
            likelihood_name: name of Likelihood.
            component_cls: class of Component.
            component_name: name of Component.

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

        print("\n" + "=" * 40)
        for key, likelihood in self.likelihoods.items():
            print(f"LIKELIHOOD {key}")
            likelihood.print_likelihood_summary(short=short)
            print("\n" + "=" * 40)

    def get_num_events_accepted(self, parameters, batch_size=1_000_000):
        """Get number of events in the histogram under given parameters.

        Args:
            batch_size: int of number of simulated events.
            parameters: dict of parameters used in simulation.

        """
        n_events = 0
        for likelihood in self.likelihoods.values():
            if hasattr(likelihood, "data_hist"):
                n_events += likelihood.get_num_events_accepted(batch_size, parameters)
            else:
                warning = f"{likelihood.name} will be omitted."
                warn(warning)
        return n_events

    def log_posterior(self, parameters, batch_size=1_000_000):
        """Get log likelihood of given parameters.

        Args:
            batch_size: int of number of simulated events.
            parameters: dict of parameters used in simulation.

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

    @property
    def _ndim(self):
        return len(self.par_manager.parameter_fit_array)

    def _set_backend(self, nwalkers=100, read_only=True, reset=False):
        if self.backend_h5 is None:
            self._backend = None
            print("With no backend")
        else:
            self._backend = emcee.backends.HDFBackend(self.backend_h5, read_only=read_only)
            if reset:
                self._backend.reset(nwalkers, self._ndim)
            print(f"With h5 backend {self.backend_h5}")

    def pre_fitting(self, nwalkers=100, read_only=True, reset=False, batch_size=1_000_000):
        """Prepare for fitting, initialize backend and sampler."""
        self._set_backend(nwalkers, read_only=read_only, reset=reset)
        self.sampler = emcee.EnsembleSampler(
            nwalkers,
            self._ndim,
            self.log_posterior,
            backend=self._backend,
            blobs_dtype=np.float32,
            parameter_names=self.par_manager.parameter_fit,
            kwargs={"batch_size": batch_size},
        )

    def fitting(self, nwalkers=200, iteration=500, batch_size=1_000_000):
        """Fitting posterior distribution of needed parameters.

        Args:
            nwalkers: int, number of walkers in the ensemble.
            iteration: int, number of steps to generate.

        """
        self._sanity_check()

        p0 = []
        for _ in range(nwalkers):
            self.par_manager.sample_init()
            p0.append(self.par_manager.parameter_fit_array)

        self.pre_fitting(nwalkers=nwalkers, read_only=False, reset=True, batch_size=batch_size)

        result = self.sampler.run_mcmc(
            p0,
            iteration,
            store=True,
            progress=True,
        )

        self._dump_meta(batch_size=batch_size)
        return result

    def continue_fitting(self, context=None, iteration=500, batch_size=1_000_000):
        """Continue a fitting of another context.

        Args:
            context: appletree context.
            iteration: int, number of steps to generate.

        """
        # If context is None, use self, i.e. continue the fitting defined in self
        if context is None:
            context = self
            p0 = None
        else:
            # Final iteration
            final_iteration = context.sampler.get_chain()[-1, :, :]
            p0 = final_iteration.tolist()

        nwalkers = context.sampler.get_chain().shape[1]

        # Init sampler for current context
        self.pre_fitting(nwalkers=nwalkers, read_only=False, reset=False, batch_size=batch_size)

        result = self.sampler.run_mcmc(
            p0,
            iteration,
            store=True,
            progress=True,
            skip_initial_state_check=True,
        )

        self._dump_meta(batch_size=batch_size)
        return result

    def get_post_parameters(self, which="mpe"):
        """Get parameters from the backend.

        Args:
            which: str, 'mpe', 'random' or 'median'. 'mpe' is the maximum posterior estimate,
            i.e. the parameter set with the highest posterior value. 'random' returns a
            random parameter set from the posterior distribution. 'median' is the marginal medians.

        """
        # Assign attributes for the first time
        # This speeds up if the user wanna call this function many times
        if not hasattr(self, "_logp"):
            self._logp = self.sampler.get_log_prob(flat=True)
        if not hasattr(self, "_chain"):
            self._chain = self.sampler.get_chain(flat=True)
        if which == "mpe":
            _parameters = self._chain[np.argmax(self._logp)]
        elif which == "random":
            _parameters = self._chain[np.random.randint(len(self._logp))]
        elif which == "median":
            _parameters = np.median(self._chain, axis=0)
        else:
            raise ValueError(f"which should be 'mpe', 'random' or 'median', got {which}!")

        _parameters = emcee.ensemble.ndarray_to_list_of_dicts(
            [_parameters],
            self.sampler.parameter_names,
        )[0]
        parameters = copy.deepcopy(self.par_manager.get_all_parameter())
        parameters.update(_parameters)
        return parameters

    def get_all_post_parameters(self, **kwargs):
        """Return all posterior parameters."""
        chain = self.sampler.get_chain(**kwargs)
        return chain

    def dump_post_parameters(self, file_name):
        """Dump max posterior parameter in .json file."""
        parameters = self.get_post_parameters()
        with open(file_name, "w") as fp:
            json.dump(parameters, fp)

    def _dump_meta(self, batch_size, metadata=None):
        """Save parameters name as attributes."""
        if metadata is None:
            metadata = {
                "version": apt.__version__,
                "date": datetime.now().strftime("%Y%m%d_%H:%M:%S"),
            }
        if self.backend_h5 is not None:
            name = self.sampler.backend.name
            with h5py.File(self.backend_h5, "r+") as opt:
                opt[name].attrs["metadata"] = json.dumps(metadata, **JSON_OPTIONS)
                # parameters prior configuration
                opt[name].attrs["par_config"] = json.dumps(
                    self.par_manager.par_config, **JSON_OPTIONS
                )
                # max posterior parameters
                opt[name].attrs["post_parameters"] = json.dumps(
                    self.get_post_parameters(), **JSON_OPTIONS
                )
                # the order of parameters saved in backend
                opt[name].attrs["parameter_fit"] = self.par_manager.parameter_fit
                # instructions
                opt[name].attrs["instruct"] = json.dumps(self.instruct, **JSON_OPTIONS)
                # configs
                opt[name].attrs["config"] = json.dumps(self.config, **JSON_OPTIONS)
                # configurations, maybe users will manually add some maps
                opt[name].attrs["_cached_configs"] = json.dumps(_cached_configs, **JSON_OPTIONS)
                # batch size
                opt[name].attrs["batch_size"] = batch_size

    def get_template(
        self,
        likelihood_name: str,
        component_name: str,
        batch_size: int = 1_000_000,
        seed: Optional[int] = None,
    ):
        """Get parameters correspondes to max posterior.

        Args:
            likelihood_name: name of Likelihood.
            component_name: name of Component.
            batch_size: int of number of simulated events.
            seed: random seed.

        """
        parameters = self.get_post_parameters()
        key = randgen.get_key(seed=seed)

        key, result = self[likelihood_name][component_name].simulate(
            key,
            batch_size,
            parameters,
        )
        return result

    def _sanity_check(self):
        """Check if needed parameters are provided."""
        needed = set(self.needed_parameters)
        provided = set(self.par_manager.get_all_parameter().keys())
        # We will not update unneeded parameters!
        if not provided.issubset(needed):
            mes = (
                f"Parameter manager should provide needed parameters only, "
                f"{sorted(provided - needed)} not needed."
            )
            raise RuntimeError(mes)

    def update_url_base(self, url_base):
        """Update url_base in appletree.share."""
        print(f"Updated url_base to {url_base}")
        set_global_config({"url_base": url_base})

    def get_parameter_config(self, par_config):
        """Get configuration for parameter manager.

        Args:
            par_config: str, parameters configuration file.

        """
        par_config = load_json(par_config)
        return par_config

    def update_parameter_config(self, likelihoods):
        needed_parameters: Set[str] = set()
        needed_rate_parameters = []
        from_parameters = []
        for likelihood in likelihoods.values():
            for k, v in likelihood["copy_parameters"].items():
                # specify rate scale
                # normalization factor, for AC & ER, etc.
                self.par_config.update({k: self.par_config[v]})
                from_parameters.append(v)
                needed_parameters.add(k)
            for k in likelihood["components"].keys():
                needed_rate_parameters.append(k + "_rate")
        for p in from_parameters:
            if p not in needed_rate_parameters and p in self.par_config:
                # Drop unused parameters
                self.par_config.pop(p)
        return needed_parameters

    @property
    def lineage(self):
        return {
            **self.instruct,
            **{
                "par_config": self.par_config,
                "likelihoods": dict(
                    zip(
                        self.likelihoods.keys(),
                        [v.lineage for v in self.likelihoods.values()],
                    )
                ),
            },
        }

    @property
    def lineage_hash(self):
        return deterministic_hash(self.lineage)
