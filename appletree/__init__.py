__version__ = '0.2.3'

# stop jax to preallocate memory
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

from . import utils
from .utils import *

from . import hist
from .hist import *

from . import interpolation
from .interpolation import *

from . import config
from .config import *

from . import parameter
from .parameter import *

from . import randgen
from .randgen import *

from . import share
from .share import *

from . import plugins

from . import plugin
from .plugin import *

from . import components
from .components import *

from . import component
from .component import *

from . import likelihood
from .likelihood import *

from . import contexts
from .contexts import *

from . import context
from .context import *

# check CUDA support setup
from warnings import warn
platform = utils.get_platform()
if platform == 'cpu':
    warning = 'You are running appletree on CPU, which usually results in low performance.'
    warn(warning)
try:
    import jax
    # try allocate something
    jax.numpy.ones(1)
except BaseException:
    if platform == 'gpu':
        print('Can not allocate memory on GPU, please check your CUDA version.')
    raise ImportError(f'Appletree is not correctly setup to be used on {platform.upper()}.')

try:
    import aptext
    HAVE_APTEXT = True
    print('Using aptext package from https://github.com/XENONnT/applefiles')
except ImportError:
    HAVE_APTEXT = False
    print('Can not find aptext')
