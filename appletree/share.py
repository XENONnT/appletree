import os
import inspect

MAPPATH = os.path.join(os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe()))), 'maps')
DATAPATH = os.path.join(os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe()))), 'data')
PARPATH = os.path.join(os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe()))), 'parameters')
CONFPATH = os.path.join(os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe()))), 'configs')

_cached_configs = {'url_base': MAPPATH}
_cached_functions = dict()
