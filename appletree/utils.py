import os
import re
import json
from warnings import warn
import pkg_resources
from time import time
from collections import namedtuple
from functools import partial

import numpy as np
import pandas as pd
import jax
from jax import numpy as jnp
from jax import jit, lax, random, vmap
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
from numpyro.distributions.util import _binomial_dispatch as _binomial_dispatch_numpyro

import GOFevaluation
from appletree.share import _cached_configs

NT_AUX_INSTALLED = False
try:
    import ntauxfiles
    NT_AUX_INSTALLED = True
except ImportError:
    pass


def exporter(export_self=False):
    """Export utility modified from https://stackoverflow.com/a/41895194
    Returns export decorator, __all__ list
    """
    all_ = []
    if export_self:
        all_.append('exporter')

    def decorator(obj):
        all_.append(obj.__name__)
        return obj

    return decorator, all_


export, __all__ = exporter(export_self=True)


@export
def use_xenon_plot_style():
    """Set matplotlib plot style."""
    params = {
        'font.family': 'serif',
        'font.size': 24, 'axes.titlesize': 24,
        'axes.labelsize': 24, 'axes.linewidth': 2,
        # ticks
        'xtick.labelsize': 22, 'ytick.labelsize': 22, 'xtick.major.size': 16, 'xtick.minor.size': 8,
        'ytick.major.size': 16, 'ytick.minor.size': 8, 'xtick.major.width': 2, 'xtick.minor.width': 2,
        'ytick.major.width': 2, 'ytick.minor.width': 2, 'xtick.direction': 'in', 'ytick.direction': 'in',
        # markers
        'lines.markersize': 12, 'lines.markeredgewidth': 3, 'errorbar.capsize': 8, 'lines.linewidth': 3,
        'savefig.bbox': 'tight', 'legend.fontsize': 24,
        'backend': 'Agg', 'mathtext.fontset': 'dejavuserif', 'legend.frameon': False,
        # figure
        'figure.facecolor': 'w',
        'figure.figsize': (12, 8),
        # pad
        'axes.labelpad': 12,
        # ticks
        'xtick.major.pad': 6, 'xtick.minor.pad': 6,
        'ytick.major.pad': 3.5, 'ytick.minor.pad': 3.5,
        # colormap
    }
    plt.rcParams.update(params)


@export
def load_data(file_name: str):
    """Load data from file. The suffix can be ".csv", ".pkl"."""
    file_name = get_file_path(file_name)
    fmt = file_name.split('.')[-1]
    if fmt == 'csv':
        data = pd.read_csv(file_name)
    elif fmt == 'pkl':
        data = pd.read_pickle(file_name)
    else:
        raise ValueError(f'unsupported file format {fmt}!')
    return data


@export
def load_json(file_name: str):
    """Load data from json file."""
    with open(get_file_path(file_name), 'r') as file:
        data = json.load(file)
    return data


@export
def camel_to_snake(x):
    """Convert x from CamelCase to snake_case,
    from https://stackoverflow.com/questions/1175208
    """
    x = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', x)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', x).lower()


@export
def _get_abspath(file_name):
    """Get the abspath of the file. Raise FileNotFoundError when not found in any subfolder"""
    for sub_dir in ('maps', 'data', 'parameters', 'configs'):
        p = os.path.join(_package_path(sub_dir), file_name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'Cannot find {file_name}')


def _package_path(sub_directory):
    """Get the abs path of the requested sub folder"""
    return pkg_resources.resource_filename('appletree', f'{sub_directory}')


@export
def get_file_path(fname):
    """Find the full path to the resource file
    Try 5 methods in the following order
    1. fname begin with '/', return absolute path
    2. can get file from _get_abspath, return appletree internal file path
    3. url_base begin with '/', return url_base + name
    4. can be found in local installed ntauxfiles, return ntauxfiles absolute path
    5. can be downloaded from MongoDB, download and return cached path
    """
    if not fname:
        warn(f'A file has value False, assuming this is intentional.')
        return

    # 1. From absolute path
    # Usually Config.default is a absolute path
    if fname.startswith('/'):
        return fname

    # 2. From appletree internal files
    try:
        return _get_abspath(fname)
    except FileNotFoundError:
        pass

    # 3. From local folder
    # Use url_base as prefix
    if 'url_base' in _cached_configs.keys():
        url_base = _cached_configs['url_base']

        if url_base.startswith('/'):
            return os.path.join(url_base, fname)

    # 4. From local installed ntauxfiles
    if NT_AUX_INSTALLED:
        # You might want to use this, for example if you are a developer
        if fname in ntauxfiles.list_private_files():
            warn(f'Using the private repo to load {fname} locally')
            fpath = ntauxfiles.get_abspath(fname)
            warn(f'Loading {fname} is successfully from {fpath}')
            return fpath

    # 5. From MongoDB
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
        warn(f'Loading {fname} from mongo downloader to {fpath}')
        return fname  # Keep the name and let get_resource do its thing

    except (FileNotFoundError, ValueError, NameError, AttributeError):
        warn(f'Mongo downloader not possible or does not have {fname}')

    warn(f'I can not find {fname}!')


@export
def timeit(indent=""):
    """Use timeit as a decorator.
    It will print out the running time of the decorated function.
    """
    def _timeit(func, indent):
        name = func.__name__

        def _func(*args, **kwargs):
            print(indent + f' Function <{name}> starts.')
            start = time()
            res = func(*args, **kwargs)
            time_ = (time() - start) * 1e3
            print(indent + f' Function <{name}> ends! Time cost = {time_:.2f} msec.')
            return res

        return _func
    if isinstance(indent, str):
        return lambda func: _timeit(func, indent)
    else:
        return _timeit(indent, "")


@export
def set_gpu_memory_usage(fraction=0.3):
    """Set GPU memory usage.
    See more on https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
    """
    if fraction > 1:
        fraction = 1
    if fraction <= 0:
        raise ValueError("fraction must be positive!")
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{fraction:.2f}"


@export
def get_equiprob_bins_2d(data,
                         n_partitions,
                         order = (0, 1),
                         x_clip = (-np.inf, +np.inf),
                         y_clip = (-np.inf, +np.inf),
                         which_np = np):
    """Get 2D equiprobable binning edges.
    :param data: array with shape (N, 2).
    :param n_partitions: [M1, M2] where M1 M2 are the number of bins on each dimension.
    :param x_clip: lower and upper binning edges on the 0th dimension.
    Data outside the x_clip will be dropped.
    :param y_clip: lower and upper binning edges on the 1st dimension.
    Data outside the y_clip will be dropped.
    :param which_np: can be numpy or jax.numpy, determining the returned array type.
    """
    mask = (data[:, 0] > x_clip[0])
    mask &= (data[:, 0] < x_clip[1])
    mask &= (data[:, 1] > y_clip[0])
    mask &= (data[:, 1] < y_clip[1])

    x_bins, y_bins = GOFevaluation.utils._get_equiprobable_binning(
        data[mask],
        n_partitions,
        order = order,
    )
    x_bins = np.clip(x_bins, *x_clip)
    y_bins = np.clip(y_bins, *y_clip)

    return which_np.array(x_bins), which_np.array(y_bins)


@export
def plot_irreg_histogram_2d(bins_x, bins_y, hist, **kwargs):
    """Plot histogram defined by irregular binning.
    :param bins_x: array with shape (M1, )
    :param bins_y: array with shape (M1-1, M2)
    :param hist: array with shape (M1-1, M2-1)
    :param density: boolean.
    """
    hist = np.asarray(hist)
    bins_x = np.asarray(bins_x)
    bins_y = np.asarray(bins_y)

    density = kwargs.get('density', False)
    cmap = mpl.cm.get_cmap("RdBu_r")

    loc = []
    width = []
    height = []
    area = []
    n = []

    for i, _ in enumerate(hist):
        for j, _ in enumerate(hist[i]):
            x_lower = bins_x[i]
            x_upper = bins_x[i+1]
            y_lower = bins_y[i, j]
            y_upper = bins_y[i, j+1]

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
            vmin=kwargs.get('vmin', np.min(n/area)),
            vmax=kwargs.get('vmax', np.max(n/area)),
            clip=False,
        )
    else:
        norm = mpl.colors.Normalize(
            vmin=kwargs.get('vmin', np.min(n)),
            vmax=kwargs.get('vmax', np.max(n)),
            clip=False,
        )

    ax = plt.subplot()
    for i, _ in enumerate(loc):
        c = n[i]/area[i] if density else n[i]
        rec = Rectangle(
            loc[i],
            width[i],
            height[i],
            facecolor=cmap(norm(c)),
            edgecolor='k',
        )
        ax.add_patch(rec)

    fig = plt.gcf()
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        label=('# events / bin size' if density else '# events'),
    )

    ax.set_xlim(np.min(bins_x), np.max(bins_x))
    ax.set_ylim(np.min(bins_y), np.max(bins_y))

    return ax


@export
def add_spaces(x):
    """Add four spaces to every line in x
    This is needed to make html raw blocks in rst format correctly
    """
    y = ''
    if isinstance(x, str):
        x = x.split('\n')
    for q in x:
        y += '    ' + q
    return y


@export
def tree_to_svg(graph_tree, save_as='data_types', view=True):
    """
    Where to save this node
    :param graph_tree: Digraph instance
    :param save_as: str, file name
    :param view: bool, Open the rendered result with the default application.
    """
    graph_tree.render(save_as, view=view)
    with open(f'{save_as}.svg', mode='r') as f:
        svg = add_spaces(f.readlines()[5:])
    os.remove(save_as)
    return svg


@export
def add_deps_to_graph_tree(context,
                           graph_tree,
                           data_names: list = ['cs1', 'cs2', 'eff'],
                           _seen = None):
    """
    Recursively add nodes to graph base on plugin.deps
    :param context: Context instance
    :param graph_tree: Digraph instance
    :param data_names: Data type name
    :param _seen: list or None, the seen data_name should not be plot
    """
    if _seen is None:
        _seen = []
    for data_name in data_names:
        if data_name in _seen:
            continue

        # Add new one
        graph_tree.node(data_name,
                        style='filled',
                        href='#' + data_name.replace('_', '-'),
                        fillcolor='white')
        if data_name == 'batch_size':
            continue
        dep_plugin = context._plugin_class_registry[data_name]
        for dep in dep_plugin.depends_on:
            graph_tree.edge(data_name, dep)
            graph_tree, _seen = add_deps_to_graph_tree(context, 
                                                       graph_tree,
                                                       dep_plugin.depends_on,
                                                       _seen)
        _seen.append(data_name)
    return graph_tree, _seen


@export
def add_plugins_to_graph_tree(context,
                              graph_tree,
                              data_names: list = ['cs1', 'cs2', 'eff'],
                              _seen = None,
                              with_data_names=False):
    """
    Recursively add nodes to graph base on plugin.deps
    :param context: Context instance
    :param graph_tree: Digraph instance
    :param data_names: Data type name
    :param _seen: list or None, the seen data_name should not be plot
    :param with_data_names: bool, whether plot even with messy data_names
    """
    if _seen is None:
        _seen = []
    for data_name in data_names:
        if data_name == 'batch_size':
            continue

        plugin = context._plugin_class_registry[data_name]
        plugin_name = plugin.__name__
        if plugin_name in _seen:
            continue

        # Add new one
        label = f'{plugin_name}'
        if with_data_names:
            label += f"\n{', '.join(plugin.depends_on)}\n{', '.join(plugin.provides)}"
        graph_tree.node(plugin_name,
                        label=label,
                        style='filled',
                        href='#' + plugin_name.replace('_', '-'),
                        fillcolor='white')

        for dep in plugin.depends_on:
            if dep == 'batch_size':
                continue
            dep_plugin = context._plugin_class_registry[dep]
            graph_tree.edge(plugin_name, dep_plugin.__name__)
            graph_tree, _seen = add_plugins_to_graph_tree(
                context,
                graph_tree,
                plugin.depends_on,
                _seen,
            )
        _seen.append(data_name)
    return graph_tree, _seen
