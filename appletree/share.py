import os
import inspect

_cached_configs = {'url_base': f'https://raw.githubusercontent.com/XENONnT/private_nt_aux_files/master/sim_files'}
_cached_functions = dict()

MAPPATH = os.path.join(os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe()))), 'maps')
DATAPATH = os.path.join(os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe()))), 'data')
PARPATH = os.path.join(os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe()))), 'parameters')
CONFPATH = os.path.join(os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe()))), 'configs')
