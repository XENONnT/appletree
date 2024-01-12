__version__ = "0.3.1"

# stop jax to preallocate memory
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from . import utils
from .utils import *

from .hist import *

from .interpolation import *

from .config import *

from .parameter import *

from .randgen import *

from .share import *

from . import plugins

from .plugin import *

from .components import *

from .component import *

from .likelihood import *

from .contexts import *

from .context import *

# check CUDA support setup
from warnings import warn

platform = utils.get_platform()
if platform == "cpu":
    warning = "You are running appletree on CPU, which usually results in low performance."
    warn(warning)
try:
    import jax

    # try allocate something
    jax.numpy.ones(1)
except BaseException:
    if platform == "gpu":
        print("Can not allocate memory on GPU, please check your CUDA version.")
    raise ImportError(f"Appletree is not correctly setup to be used on {platform.upper()}.")

try:
    import aptext

    HAVE_APTEXT = True
    print("Using aptext package from https://github.com/XENONnT/applefiles")
except ImportError:
    HAVE_APTEXT = False
    print("Can not find aptext")
