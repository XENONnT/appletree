import numpy as np
import appletree as apt
import inspect
import h5py

from typing import Set
from appletree.share import _cached_functions


class Transformer:
    domain: Set[str] = set()
    codomain: Set[str] = set()

    def transform(self, param):
        raise NotImplementedError

    def inverse_transform(self, param):
        raise NotImplementedError

    def jacobian(self, param):
        raise NotImplementedError

    def wrapped_transform(self, param):
        res = {key: value for key, value in param.items() if key not in self.domain}
        res.update(self.transform(param))
        return res

    def wrapped_inverse_transform(self, param):
        res = {key: value for key, value in param.items() if key not in self.codomain}
        res.update(self.inverse_transform(param))
        return res

    @staticmethod
    def _trans_param(param_arg=-1, transform=lambda x: x):
        def decorator(func):
            def wrapper(*args, **kwargs):
                if len(args) > param_arg:
                    args_list = list(args)
                    args_list[param_arg] = transform(args_list[param_arg])
                    args = tuple(args_list)
                else:
                    kwargs["parameters"] = transform(kwargs["parameters"])
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def trans_param_arg(self, param_arg):
        return self._trans_param(param_arg, self.wrapped_transform)

    def inv_trans_param_arg(self, param_arg):
        return self._trans_param(param_arg, self.wrapped_inverse_transform)

    def __call__(self, obj):
        if isinstance(obj, dict):
            return self.wrapped_transform(obj)
        elif issubclass(obj, apt.Parameter):
            return get_transformed_parameter_class(self)
        elif issubclass(obj, apt.Component):
            return get_transformed_component_class(self, obj)
        elif issubclass(obj, apt.Likelihood):
            return get_transformed_likelihood_class(self)
        elif issubclass(obj, apt.Context):
            return get_transformed_context_class(self, obj)


def get_transformed_component_class(transformer, component_class):
    inv_trans_param_arg = transformer.inv_trans_param_arg
    domain = transformer.domain
    codomain = transformer.codomain

    class TransformedComponent(component_class):
        def compile(self):
            if hasattr(self, "_compile"):
                self._compile()
            else:  # It's a ComponentFixed
                pass

        @inv_trans_param_arg(param_arg=3)
        def simulate(self, key, batch_size, parameters):
            return _cached_functions[self.llh_name][self.func_name](key, batch_size, parameters)

        @inv_trans_param_arg(param_arg=2)
        def get_normalization(self, hist, parameters, batch_size=None):
            return super().get_normalization(hist, parameters, batch_size)

        @property
        def all_parameters(self):
            if not super().all_parameters & domain:
                return self._all_parameters
            else:
                return (self._all_parameters - domain) | codomain

        @property
        def needed_parameters(self):
            if not super().needed_parameters & domain:
                return self._needed_parameters
            else:
                return (self._needed_parameters - domain) | codomain

    return TransformedComponent


def get_transformed_parameter_class(transformer):
    transform = transformer.wrapped_transform
    inverse_transform = transformer.wrapped_inverse_transform
    domain = transformer.domain
    codomain = transformer.codomain

    class TransformedParameter(apt.Parameter):
        @property
        def parameter_fit(self):
            parameter_fit = set(super().parameter_fit)
            if not parameter_fit & domain:
                return sorted(parameter_fit)
            else:
                return sorted((parameter_fit - domain) | codomain)

        @property
        def parameter_all(self):
            parameter_all = set(super().parameter_all)
            if not parameter_all & domain:
                return sorted(parameter_all)
            else:
                return sorted((parameter_all - domain) | codomain)

        @property
        def parameter_fixed(self):
            return sorted(set(self.parameter_all) - set(self.parameter_fit))

        def set_parameter(self, keys, vals=None):
            if isinstance(keys, list):
                assert len(keys) == len(vals), "keys must have the same length as vals!"
                parameter = inverse_transform({key: val for key, val in zip(keys, vals)})
                super().set_parameter(parameter)
            elif isinstance(keys, dict):
                parameter = inverse_transform(keys)
                super().set_parameter(parameter)
            elif isinstance(keys, str):
                self.set_parameter({keys: vals})
            else:
                raise ValueError("keys must be a str or a list of str!")

        def __getitem__(self, keys):
            """__getitem__, keys can be str/list/set."""
            if isinstance(keys, (set, list)):
                return np.array([transform(self._parameter_dict)[key] for key in keys])
            elif isinstance(keys, str):
                return transform(self._parameter_dict)[keys]
            else:
                raise ValueError("keys must be a str or a list of str!")

        def get_all_parameter(self):
            """Return all parameters as a dict."""
            return transform(self._parameter_dict)

        @property
        def log_prior(self):
            res = super().log_prior
            res += np.log(transformer.jacobian(self._parameter_dict))
            return res

    return TransformedParameter


def get_transformed_likelihood_class(transformer):
    class TransformedLikelihood(apt.Likelihood):
        def register_component(self, component_cls, *args, **kwargs):
            super().register_component(transformer(component_cls), *args, **kwargs)

    return TransformedLikelihood


def get_transformed_context_class(transformer, context_class):
    class TransformedContext(context_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            if self.needed_parameters & transformer.domain:
                self.needed_parameters = (
                    self.needed_parameters - transformer.domain
                ) | transformer.codomain
            self.par_manager = transformer(apt.Parameter)(self.par_config)

        def register_likelihood(self, likelihood_name, likelihood_config):
            if likelihood_name in self.likelihoods:
                raise ValueError(f"Likelihood named {likelihood_name} already existed!")
            likelihood = getattr(apt, likelihood_config.get("type", "Likelihood"))
            self.likelihoods[likelihood_name] = transformer(likelihood)(
                name=likelihood_name,
                **likelihood_config,
            )

        def _dump_meta(self, metadata=None):
            super()._dump_meta(metadata)
            if self.backend_h5 is not None:
                name = self.sampler.backend.name
                try:
                    transformer_src = inspect.getsource(transformer.__class__)
                except Exception as e:
                    print(f"Failed to get source code of transformer: {e}")
                    transformer_src = transformer.__repr__()
                with h5py.File(self.backend_h5, "r+") as opt:
                    opt[name].attrs["transformer"] = transformer_src

    return TransformedContext
