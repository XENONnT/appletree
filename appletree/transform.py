

component_class = apt.ERBand
domain = {'g1', 'g2'}
codomain = {'a', 'b'}

def transform(param):
    res = {key: value for key, value in param.items() if key not in domain}
    res.update({'a': param['g1'] * 0.5,
                'b': param['g2'] * 2,})
    return res
    
def inverse_transform(param):
    res = {key: value for key, value in param.items() if key not in codomain}
    res.update({'g1': param['a'] * 2,
                'g2': param['b'] * 0.5,})
    return res

def trans_param(param_arg=-1, transform=lambda x: x):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if len(args) > param_arg:
                args_list = list(args)
                args_list[param_arg] = transform(args_list[param_arg])
                args = tuple(args_list)
                return func(*args, **kwargs)
            else:
                kwargs['parameters'] = transform(kwargs['parameters'])
        return wrapper
    return decorator


class TransformedComponentSim(component_class):
    def compile(self):
        self._compile()

    @trans_param(param_arg=3, transform=inverse_transform)
    def simulate(self, key, batch_size, parameters):
        return _cached_functions[self.llh_name][self.func_name](key, batch_size, parameters)

    @trans_param(param_arg=2, transform=inverse_transform)
    def get_normalization(self, hist, parameters, batch_size=None):
        return super().get_normalization(self, hist, parameters, batch_size=None)

    @property
    def all_parameters(self):
        if not self._all_parameters >= domain:
            raise ValueError("The domain of a transfomer must be the subset of all_parameters of original component!")
        return (self._all_parameters - domain) | codomain

    @property
    def needed_parameters(self):
        if not self._needed_parameters >= domain:
            raise ValueError("The domain of a transfomer must be the subset of needed_parameters of original component!")
        return (self._needed_parameters - domain) | codomain
