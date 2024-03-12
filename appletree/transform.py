import appletree as apt

from appletree.share import _cached_functions


class Transformer:
    domain = {"g1", "g2"}
    codomain = {"a", "b"}

    def transform(self, param):
        res = {key: value for key, value in param.items() if key not in self.domain}
        res.update(
            {
                "a": param["g1"] * 0.5,
                "b": param["g2"] * 2,
            }
        )
        return res

    def inverse_transform(self, param):
        res = {key: value for key, value in param.items() if key not in self.codomain}
        res.update(
            {
                "g1": param["a"] * 2,
                "g2": param["b"] * 0.5,
            }
        )
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
        return self._trans_param(param_arg, self.transform)

    def inv_trans_param_arg(self, param_arg):
        return self._trans_param(param_arg, self.inverse_transform)

    def __call__(self, obj):
        if isinstance(obj, dict):
            return self.transform(obj)
        elif issubclass(obj, apt.Parameter):
            return get_transformed_parameter_class(self.transform, self.domain, self.codomain)
        elif issubclass(obj, apt.Component):
            return get_transformed_component_class(
                obj, self.trans_param_arg, self.inv_trans_param_arg, self.domain, self.codomain
            )
        elif issubclass(obj, apt.Likelihood):
            return get_transformed_likelihood_class(self)


def get_transformed_component_class(
    component_class, trans_param_arg, inv_trans_param_arg, domain, codomain
):
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
            return super().get_normalization(self, hist, parameters, batch_size=None)

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


def get_transformed_parameter_class(transform, domain, codomain):
    class TransformedParameter(apt.Parameter):
        @property
        def parameter_fit(self):
            if not super().parameter_fit & domain:
                return self._parameter_fit
            else:
                return (self._parameter_fit - domain) | codomain

        @property
        def parameter_all(self):
            if not super().parameter_all & domain:
                return self._parameter_all
            else:
                return (self._parameter_all - domain) | codomain

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

    return TransformedParameter


def get_transformed_likelihood_class(transformer):
    class TransformedLikelihood(apt.Likelihood):
        def register_component(self, component_cls, *args, **kwargs):
            super().register_component(transformer(component_cls), *args, **kwargs)

    return TransformedLikelihood
