import os
import json
from warnings import warn
import importlib_resources
from time import time

from jax.lib import xla_bridge
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
from scipy.special import erf
from scipy.optimize import root
from scipy.stats import chi2

import GOFevaluation
from appletree.share import _cached_configs

NT_AUX_INSTALLED = False
try:
    import ntauxfiles

    NT_AUX_INSTALLED = True
except ImportError:
    pass

SKIP_MONGO_DB = True


def exporter(export_self=False):
    """Export utility modified from https://stackoverflow.com/a/41895194
    Returns export decorator, __all__ list
    """
    all_ = []
    if export_self:
        all_.append("exporter")

    def decorator(obj):
        all_.append(obj.__name__)
        return obj

    return decorator, all_


export, __all__ = exporter(export_self=True)


@export
def use_xenon_plot_style():
    """Set matplotlib plot style."""
    params = {
        "font.family": "serif",
        "font.size": 24,
        "axes.titlesize": 24,
        "axes.labelsize": 24,
        "axes.linewidth": 2,
        # ticks
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "xtick.major.size": 16,
        "xtick.minor.size": 8,
        "ytick.major.size": 16,
        "ytick.minor.size": 8,
        "xtick.major.width": 2,
        "xtick.minor.width": 2,
        "ytick.major.width": 2,
        "ytick.minor.width": 2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        # markers
        "lines.markersize": 12,
        "lines.markeredgewidth": 3,
        "errorbar.capsize": 8,
        "lines.linewidth": 3,
        "savefig.bbox": "tight",
        "legend.fontsize": 24,
        "backend": "Agg",
        "mathtext.fontset": "dejavuserif",
        "legend.frameon": False,
        # figure
        "figure.facecolor": "w",
        "figure.figsize": (12, 8),
        # pad
        "axes.labelpad": 12,
        # ticks
        "xtick.major.pad": 6,
        "xtick.minor.pad": 6,
        "ytick.major.pad": 3.5,
        "ytick.minor.pad": 3.5,
        # colormap
    }
    plt.rcParams.update(params)


@export
def load_data(file_name: str):
    """Load data from file.

    The suffix can be ".csv", ".pkl".

    """
    file_name = get_file_path(file_name)
    fmt = file_name.split(".")[-1]
    if fmt == "csv":
        data = pd.read_csv(file_name)
    elif fmt == "pkl":
        data = pd.read_pickle(file_name)
    else:
        raise ValueError(f"unsupported file format {fmt}!")
    return data


@export
def load_json(file_name: str):
    """Load data from json file."""
    with open(get_file_path(file_name), "r") as file:
        data = json.load(file)
    return data


@export
def _get_abspath(file_name):
    """Get the abspath of the file.

    Raise FileNotFoundError when not found in any subfolder

    """
    for sub_dir in ("maps", "data", "parameters", "instructs"):
        p = os.path.join(_package_path(sub_dir), file_name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find {file_name}")


def _package_path(sub_directory):
    """Get the abs path of the requested sub folder."""
    return importlib_resources.files("appletree") / sub_directory


@export
def get_file_path(fname):
    """Find the full path to the resource file. Try 5 methods in the following order.

    * fname begin with '/', return absolute path
    * url_base begin with '/', return url_base + name
    * can get file from _get_abspath, return appletree internal file path
    * can be found in local installed ntauxfiles, return ntauxfiles absolute path
    * can be downloaded from MongoDB, download and return cached path

    """
    # 1. From absolute path if file exists
    # Usually Config.default is a absolute path
    if os.path.isfile(fname):
        return fname

    # 2. From local folder
    # Use url_base as prefix
    if "url_base" in _cached_configs.keys():
        url_base = _cached_configs["url_base"]

        if url_base.startswith("/"):
            fpath = os.path.join(url_base, fname)
            if os.path.exists(fpath):
                warn(f"Load {fname} successfully from {fpath}")
                return fpath

    # 3. From appletree internal files
    try:
        return _get_abspath(fname)
    except FileNotFoundError:
        pass

    # 4. From local installed ntauxfiles
    if NT_AUX_INSTALLED:
        # You might want to use this, for example if you are a developer
        if fname in ntauxfiles.list_private_files():
            fpath = ntauxfiles.get_abspath(fname)
            warn(f"Load {fname} successfully from {fpath}")
            return fpath

    # 5. From MongoDB
    if not SKIP_MONGO_DB:
        try:
            import straxen

            # https://straxen.readthedocs.io/en/latest/config_storage.html
            # downloading-xenonnt-files-from-the-database  # noqa

            # we need to add the straxen.MongoDownloader() in this
            # try: except NameError: logic because the NameError
            # gets raised if we don't have access to utilix.
            downloader = straxen.MongoDownloader()
            # FileNotFoundError, ValueErrors can be raised if we
            # cannot load the requested config
            fpath = downloader.download_single(fname)
            warn(f"Loading {fname} from mongo downloader to {fpath}")
            return fname  # Keep the name and let get_resource do its thing
        except (FileNotFoundError, ValueError, NameError, AttributeError):
            warn(f"Mongo downloader not possible or does not have {fname}")

    # raise error when can not find corresponding file
    raise RuntimeError(f"Can not find {fname}, please check your file system")


@export
def timeit(indent=""):
    """Use timeit as a decorator.

    It will print out the running time of the decorated function.

    """

    def _timeit(func, indent):
        name = func.__name__

        def _func(*args, **kwargs):
            print(indent + f" Function <{name}> starts.")
            start = time()
            res = func(*args, **kwargs)
            time_ = (time() - start) * 1e3
            print(indent + f" Function <{name}> ends! Time cost = {time_:.2f} msec.")
            return res

        return _func

    if isinstance(indent, str):
        return lambda func: _timeit(func, indent)
    else:
        return _timeit(indent, "")


@export
def get_platform():
    """Show the platform we are using, either cpu ot gpu."""
    return xla_bridge.get_backend().platform


@export
def set_gpu_memory_usage(fraction=0.3):
    """Set GPU memory usage.

    See more on https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html

    """
    if fraction > 1:
        fraction = 0.99
    if fraction <= 0:
        raise ValueError("fraction must be positive!")
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f".{int(fraction * 100):d}"


@export
def get_equiprob_bins_1d(
    data,
    n_partitions,
    clip=(-np.inf, +np.inf),
    which_np=np,
):
    """Get 2D equiprobable binning edges.

    Args:
        data: array with shape N.
        n_partitions: M1 which is the number of bins.
        clip: lower and upper binning edges on the 0th dimension.
            Data outside the clip will be dropped.
        Data outside the y_clip will be dropped.
        which_np: can be numpy or jax.numpy, determining the returned array type.

    """
    mask = data > clip[0]
    mask &= data < clip[1]

    bins = GOFevaluation.utils.get_equiprobable_binning(
        data[mask],
        n_partitions,
    )
    # To be strict, clip the inf(s)
    bins = np.clip(bins, *clip)

    return which_np.array(bins)


@export
def get_equiprob_bins_2d(
    data,
    n_partitions,
    order=(0, 1),
    x_clip=(-np.inf, +np.inf),
    y_clip=(-np.inf, +np.inf),
    which_np=np,
):
    """Get 2D equiprobable binning edges.

    Args:
        data: array with shape (N, 2).
        n_partitions: [M1, M2] where M1 M2 are the number of bins on each dimension.
        x_clip: lower and upper binning edges on the 0th dimension.
            Data outside the x_clip will be dropped.
        y_clip: lower and upper binning edges on the 1st dimension.
        Data outside the y_clip will be dropped.
        which_np: can be numpy or jax.numpy, determining the returned array type.

    """
    mask = data[:, 0] > x_clip[0]
    mask &= data[:, 0] < x_clip[1]
    mask &= data[:, 1] > y_clip[0]
    mask &= data[:, 1] < y_clip[1]

    x_bins, y_bins = GOFevaluation.utils.get_equiprobable_binning(
        data[mask],
        n_partitions,
        order=order,
    )
    # To be strict, clip the inf(s)
    x_bins = np.clip(x_bins, *x_clip)
    y_bins = np.clip(y_bins, *y_clip)

    return which_np.array(x_bins), which_np.array(y_bins)


@export
def plot_irreg_histogram_2d(bins_x, bins_y, hist, **kwargs):
    """Plot histogram defined by irregular binning.

    Args:
        bins_x: array with shape (M1,).
        bins_y: array with shape (M1-1, M2).
        hist: array with shape (M1-1, M2-1).
        density: boolean.

    """
    hist = np.asarray(hist)
    bins_x = np.asarray(bins_x)
    bins_y = np.asarray(bins_y)

    density = kwargs.get("density", False)
    cmap = mpl.cm.RdBu_r

    loc = []
    width = []
    height = []
    area = []
    n = []

    for i, _ in enumerate(hist):
        for j, _ in enumerate(hist[i]):
            x_lower = bins_x[i]
            x_upper = bins_x[i + 1]
            y_lower = bins_y[i, j]
            y_upper = bins_y[i, j + 1]

            loc.append((x_lower, y_lower))
            width.append(x_upper - x_lower)
            height.append(y_upper - y_lower)
            area.append((x_upper - x_lower) * (y_upper - y_lower))
            n.append(hist[i, j])

    loc = np.asarray(loc)
    width = np.asarray(width)
    height = np.asarray(height)
    area = np.asarray(area)
    n = np.asarray(n)

    if density:
        norm = mpl.colors.Normalize(
            vmin=kwargs.get("vmin", np.min(n / area)),
            vmax=kwargs.get("vmax", np.max(n / area)),
            clip=False,
        )
    else:
        norm = mpl.colors.Normalize(
            vmin=kwargs.get("vmin", np.min(n)),
            vmax=kwargs.get("vmax", np.max(n)),
            clip=False,
        )

    ax = plt.subplot()
    for i, _ in enumerate(loc):
        c = n[i] / area[i] if density else n[i]
        rec = Rectangle(
            loc[i],
            width[i],
            height[i],
            facecolor=cmap(norm(c)),
            edgecolor="k",
        )
        ax.add_patch(rec)

    fig = plt.gcf()
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        label=("# events / bin size" if density else "# events"),
    )

    ax.set_xlim(np.min(bins_x), np.max(bins_x))
    ax.set_ylim(np.min(bins_y), np.max(bins_y))

    return ax


@export
def add_spaces(x):
    """Add four spaces to every line in x This is needed to make html raw blocks in rst format
    correctly."""
    y = ""
    if isinstance(x, str):
        x = x.split("\n")
    for q in x:
        y += "    " + q
    return y


@export
def tree_to_svg(graph_tree, save_as="data_types", view=True):
    """Where to save this node.

    Args:
        graph_tree: Digraph instance.
        save_as: str, file name.
        view: bool, Open the rendered result with the default application.

    """
    graph_tree.render(save_as, view=view)
    with open(f"{save_as}.svg", mode="r") as f:
        svg = add_spaces(f.readlines()[5:])
    os.remove(save_as)
    return svg


@export
def add_deps_to_graph_tree(
    component, graph_tree, data_names: list = ["cs1", "cs2", "eff"], _seen=None
):
    """Recursively add nodes to graph base on plugin.deps.

    Args:
        context: Context instance.
        graph_tree: Digraph instance.
        data_names: Data type name.
        _seen: list or None, the seen data_name should not be plot.

    """
    if _seen is None:
        _seen = []
    for data_name in data_names:
        if data_name in _seen:
            continue

        # Add new one
        graph_tree.node(
            data_name, style="filled", href="#" + data_name.replace("_", "-"), fillcolor="white"
        )
        if data_name == "batch_size":
            continue
        dep_plugin = component._plugin_class_registry[data_name]
        for dep in dep_plugin.depends_on:
            graph_tree.edge(data_name, dep)
            graph_tree, _seen = add_deps_to_graph_tree(
                component,
                graph_tree,
                dep_plugin.depends_on,
                _seen,
            )
        _seen.append(data_name)
    return graph_tree, _seen


@export
def add_plugins_to_graph_tree(
    component,
    graph_tree,
    data_names: list = ["cs1", "cs2", "eff"],
    _seen=None,
    with_data_names=False,
):
    """Recursively add nodes to graph base on plugin.deps.

    Args:
        context: Context instance.
        graph_tree: Digraph instance.
        data_names: Data type name.
        _seen: list or None, the seen data_name should not be plot.
        with_data_names: bool, whether plot even with messy data_names

    """
    if _seen is None:
        _seen = []
    for data_name in data_names:
        if data_name == "batch_size":
            continue

        plugin = component._plugin_class_registry[data_name]
        plugin_name = plugin.__name__
        if plugin_name in _seen:
            continue

        # Add new one
        label = f"{plugin_name}"
        if with_data_names:
            label += f"\n{', '.join(plugin.depends_on)}\n{', '.join(plugin.provides)}"
        graph_tree.node(
            plugin_name,
            label=label,
            style="filled",
            href="#" + plugin_name.replace("_", "-"),
            fillcolor="white",
        )

        for dep in plugin.depends_on:
            if dep == "batch_size":
                continue
            dep_plugin = component._plugin_class_registry[dep]
            graph_tree.edge(plugin_name, dep_plugin.__name__)
            graph_tree, _seen = add_plugins_to_graph_tree(
                component,
                graph_tree,
                plugin.depends_on,
                _seen,
            )
        _seen.append(plugin_name)
    return graph_tree, _seen


@export
def add_extensions(module1, module2, base, force=False):
    """Add subclasses of module2 to module1.

    When ComponentSim compiles the dependency tree, it will search in the appletree.plugins module
    for Plugin(as attributes). When building Likelihood, it will also search for corresponding
    Component(s) specified in the instructions(e.g. NRBand).

    So we need to assign the attributes before compilation. These plugins are mostly user defined.

    """
    # Assign the module2 as attribute of module1
    is_exists = module2.__name__ in dir(module1)
    if is_exists and not force:
        raise ValueError(
            f"{module2.__name__} already existed in {module1.__name__}, "
            f"do not re-register a module with same name",
        )
    else:
        if is_exists:
            print(f"You have forcibly registered {module2.__name__} to {module1.__name__}")
        setattr(module1, module2.__name__.split(".")[-1], module2)

    # Iterate the module2 and assign the single Plugin(s) as attribute(s)
    for x in dir(module2):
        x = getattr(module2, x)
        if not isinstance(x, type(type)):
            continue
        _add_extension(module1, x, base, force=force)


def _add_extension(module, subclass, base, force=False):
    """Add subclass to module Skip the class when it is base class.

    It is no allowed to assign a class which has same name to an already assigned class. We do not
    allowed class name covering! Please change the name of your class when Error shows itself.

    """
    if getattr(subclass, "_" + subclass.__name__ + "__is_base", False):
        return

    if issubclass(subclass, base) and subclass != base:
        is_exists = subclass.__name__ in dir(module)
        if is_exists and not force:
            raise ValueError(
                f"{subclass.__name__} already existed in {module.__name__}, "
                f"do not re-register a {base.__name__} with same name",
            )
        else:
            if is_exists:
                print(f"You have forcibly registered {subclass.__name__} to {module.__name__}")
            setattr(module, subclass.__name__, subclass)


def integrate_midpoint(x, y):
    """Calculate the integral using midpoint method.

    Args:
        x: 1D array-like.
        y: 1D array-like, with the same length as x.

    """
    _, res = cum_integrate_midpoint(x, y)
    return res[-1]


def cum_integrate_midpoint(x, y):
    """Calculate the cumulative integral using midpoint method.

    Args:
        x: 1D array-like.
        y: 1D array-like, with the same length as x.

    """
    x = np.array(x)
    y = np.array(y)
    dx = x[1:] - x[:-1]
    x_mid = 0.5 * (x[1:] + x[:-1])
    y_mid = 0.5 * (y[1:] + y[:-1])
    return x_mid, np.cumsum(dx * y_mid)


@export
def check_unused_configs():
    """Check if there are unused configs."""
    unused_configs = set(_cached_configs.keys()) - _cached_configs.accessed_keys
    if unused_configs:
        warn(f"Detected unused configs: {unused_configs}, you might set the configs incorrectly.")


@export
def errors_to_two_half_norm_sigmas(errors):
    """This function solves the sigmas for a two-half-norm distribution, such that the 16 and 84
    percentile corresponds to the given errors.

    In the two-half-norm distribution, the positive and negative errors are assumed to be
    the std of the glued normal distributions. While we interpret the 16 and 84 percentile as
    the input errors, thus we need to solve the sigmas for the two-half-norm distribution.
    The solution is determined by the following conditions:
    - The 16 percentile of the two-half-norm distribution should be the negative error.
    - The 84 percentile of the two-half-norm distribution should be the positive error.
    - The mode of the two-half-norm distribution should be 0.

    """

    def _to_solve(x, errors, p):
        return [
            x[0] / (x[0] + x[1]) * (1 - erf(errors[0] / x[0] / np.sqrt(2))) - p / 2,
            x[1] / (x[0] + x[1]) * (1 - erf(errors[1] / x[1] / np.sqrt(2))) - p / 2,
        ]

    res = root(_to_solve, errors, args=(errors, 1 - chi2.cdf(1, 1)))
    assert res.success, f"Cannot solve sigmas of TwoHalfNorm for errors {errors}!"
    return res.x
