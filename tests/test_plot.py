import appletree as apt
from appletree import Plotter
from appletree.plot import (
    plot_maps, _collect_maps, _collapse_regbin_map,
)
from appletree.share import _cached_configs, _cached_functions
from appletree.utils import load_json
import numpy as np


def test_plot():
    """Test plot of Rn220 fitting."""
    instruction = load_json("rn220.json")

    filename = "rn220.h5"
    instruction["backend_h5"] = filename

    context = apt.Context(instruction)

    context.print_context_summary(short=False)
    context.fitting(nwalkers=100, iteration=2, batch_size=int(1e4))

    plotter = Plotter(filename)
    plotter.make_all_plots()


def test_plot_maps():
    """Test plot_maps with a Rn220 context."""
    _cached_functions.clear()
    _cached_configs.clear()
    context = apt.ContextRn220()

    figures = plot_maps(context)
    assert len(figures) > 0
    for fig, ax in figures:
        assert fig is not None


def test_plot_maps_collapse():
    """Test plot_maps with collapse argument for 3D maps."""
    _cached_functions.clear()
    _cached_configs.clear()
    context = apt.ContextRn220()

    figures = plot_maps(context, collapse={"s1_correction": {"z": None}})
    assert len(figures) > 0
    for fig, ax in figures:
        assert fig is not None


def test_plot_maps_collapse_range():
    """Test plot_maps with collapse and range selection."""
    _cached_functions.clear()
    _cached_configs.clear()
    context = apt.ContextRn220()

    figures = plot_maps(context, collapse={"s1_correction": {"z": (-100, -50)}})
    assert len(figures) > 0
    for fig, ax in figures:
        assert fig is not None


def test_collect_maps_deduplication():
    """Test that _collect_maps deduplicates maps used by multiple plugins."""
    from appletree.config import Map, SigmaMap
    from appletree.component import ComponentSim

    _cached_functions.clear()
    _cached_configs.clear()
    context = apt.ContextRn220Ar37()

    collected = _collect_maps(context)
    assert len(collected) > 0

    # Count total map configs across all plugins (with duplicates)
    total = 0
    for likelihood in context.likelihoods.values():
        for component in likelihood.components.values():
            if not isinstance(component, ComponentSim):
                continue
            for work in component.worksheet:
                provides = work[1]
                plugin_class = component._plugin_class_registry[provides[0]]
                for config in plugin_class.takes_config.values():
                    if isinstance(config, (Map, SigmaMap)):
                        total += 1

    # The combined context shares maps across likelihoods,
    # so the deduplicated count must be strictly less.
    assert len(collected) < total


def test_collapse_regbin_map():
    """Test _collapse_regbin_map with a synthetic 3D map."""
    map_data = np.arange(24).reshape(2, 3, 4).astype(float)
    coord_name = ["x", "y", "z"]
    coord_lowers = [0.0, 0.0, 0.0]
    coord_uppers = [2.0, 3.0, 4.0]

    # Collapse z axis
    result, names, lowers, uppers = _collapse_regbin_map(
        map_data, coord_name, coord_lowers, coord_uppers,
        collapse={"z": None},
    )
    assert result.shape == (2, 3)
    assert names == ["x", "y"]

    # Collapse with range
    result, names, lowers, uppers = _collapse_regbin_map(
        map_data, coord_name, coord_lowers, coord_uppers,
        collapse={"z": (0.5, 1.5)},
    )
    assert result.shape == (2, 3)
    assert names == ["x", "y"]

    # Collapse two axes
    result, names, lowers, uppers = _collapse_regbin_map(
        map_data, coord_name, coord_lowers, coord_uppers,
        collapse={"y": None, "z": None},
    )
    assert result.shape == (2,)
    assert names == ["x"]

    # No collapse
    result, names, lowers, uppers = _collapse_regbin_map(
        map_data, coord_name, coord_lowers, coord_uppers,
        collapse=None,
    )
    assert result.shape == (2, 3, 4)
    assert names == ["x", "y", "z"]
