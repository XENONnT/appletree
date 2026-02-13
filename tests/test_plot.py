from types import SimpleNamespace

import appletree as apt
from appletree import Plotter
from appletree.plot import (
    plot_maps, _collect_maps, _collapse_regbin_map, _regbin_edges_centers,
    _plot_map_1d_point, _plot_map_1d_regbin, _plot_map_2d_regbin,
    _plot_sigma_map_1d_regbin,
    _plot_regular_map, _plot_sigma_map,
)
from appletree.share import _cached_configs, _cached_functions
from appletree.utils import load_json
import numpy as np


def _make_mock_sigma(coord_name, coord_lowers, coord_uppers, shape):
    """Create a mock SigmaMap-like object."""
    inner = SimpleNamespace(
        coordinate_type="regbin",
        coordinate_name=coord_name,
        coordinate_lowers=coord_lowers,
        coordinate_uppers=coord_uppers,
        map=np.ones(shape),
    )
    return SimpleNamespace(
        name="test_sigma",
        median=inner,
        lower=SimpleNamespace(map=np.zeros(shape)),
        upper=SimpleNamespace(map=np.ones(shape) * 2),
    )


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


def test_plot_maps(tmp_path):
    """Test plot_maps with various collapse modes and saving."""
    _cached_functions.clear()
    _cached_configs.clear()
    context = apt.ContextRn220()

    # No collapse
    figures = plot_maps(context)
    assert len(figures) > 0
    for fig, ax in figures:
        assert fig is not None

    # Collapse 3D map
    figures = plot_maps(context, collapse={"s1_correction": {"z": None}})
    assert len(figures) > 0

    # Collapse with range
    figures = plot_maps(context, collapse={"s1_correction": {"z": (-100, -50)}})
    assert len(figures) > 0

    # Save to disk
    import os
    save_dir = str(tmp_path / "maps")
    figures = plot_maps(
        context,
        collapse={"s1_correction": {"z": None}},
        save=True,
        save_path=save_dir,
        fmt="png",
    )
    assert len(os.listdir(save_dir)) == len(figures)


def test_collect_maps_deduplication():
    """Test that _collect_maps deduplicates maps across likelihoods."""
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
    """Test _collapse_regbin_map with various collapse modes."""
    map_data = np.arange(24).reshape(2, 3, 4).astype(float)
    args = (["x", "y", "z"], [0.0, 0.0, 0.0], [2.0, 3.0, 4.0])

    # Collapse z axis
    result, names, _, _ = _collapse_regbin_map(map_data, *args, collapse={"z": None})
    assert result.shape == (2, 3)
    assert names == ["x", "y"]

    # Collapse with range
    result, names, _, _ = _collapse_regbin_map(
        map_data, *args, collapse={"z": (0.5, 1.5)},
    )
    assert result.shape == (2, 3)
    assert names == ["x", "y"]

    # Collapse two axes
    result, names, _, _ = _collapse_regbin_map(
        map_data, *args, collapse={"y": None, "z": None},
    )
    assert result.shape == (2,)
    assert names == ["x"]

    # No collapse
    result, names, _, _ = _collapse_regbin_map(map_data, *args, collapse=None)
    assert result.shape == (2, 3, 4)

    # Range matching no bins falls back to all bins
    map_2d = np.arange(6).reshape(2, 3).astype(float)
    result, names, _, _ = _collapse_regbin_map(
        map_2d, ["x", "y"], [0.0, 0.0], [2.0, 3.0],
        collapse={"y": (100.0, 200.0)},
    )
    assert result.shape == (2,)


def test_regbin_edges_centers():
    """Test _regbin_edges_centers for linear and log cases."""
    edges, centers = _regbin_edges_centers(1.0, 1000.0, 3, is_log=True)
    np.testing.assert_allclose(edges, [1, 10, 100, 1000])
    np.testing.assert_allclose(
        centers, [np.sqrt(10), np.sqrt(1000), np.sqrt(100000)],
    )

    edges, centers = _regbin_edges_centers(0.0, 3.0, 3, is_log=False)
    np.testing.assert_allclose(edges, [0, 1, 2, 3])
    np.testing.assert_allclose(centers, [0.5, 1.5, 2.5])


def test_plot_helpers():
    """Test individual plotting helpers with mock data."""
    # 1D point map
    point_map = SimpleNamespace(
        name="test", coordinate_type="point",
        coordinate_name="x",
        coordinate_system=[1.0, 2.0, 3.0], map=[0.5, 1.0, 1.5],
    )
    fig, _ = _plot_map_1d_point(point_map)
    assert fig is not None

    # 1D regbin map
    fig, _ = _plot_map_1d_regbin(
        point_map, np.array([1.0, 2.0, 3.0]), "x", 0.0, 3.0,
    )
    assert fig is not None

    # 2D log regbin map
    map_2d = SimpleNamespace(
        name="test_2d", coordinate_type="log_regbin",
        coordinate_name=["x", "y"],
        coordinate_lowers=[1.0, 1.0], coordinate_uppers=[100.0, 100.0],
        map=np.ones((3, 3)),
    )
    fig, _ = _plot_map_2d_regbin(
        map_2d, np.ones((3, 3)),
        ["x", "y"], [1.0, 1.0], [100.0, 100.0], is_log=True,
    )
    assert fig is not None

    # 1D regbin sigma map
    sigma_1d = _make_mock_sigma(["x"], [0.0], [3.0], (3,))
    fig, _ = _plot_sigma_map_1d_regbin(
        sigma_1d,
        np.array([1.0, 2.0, 3.0]),
        np.array([0.5, 1.5, 2.5]),
        np.array([1.5, 2.5, 3.5]),
        "x", 0.0, 3.0,
    )
    assert fig is not None


def test_plot_routing():
    """Test _plot_regular_map and _plot_sigma_map routing and edge cases."""
    # Unknown coordinate type returns None
    assert _plot_regular_map(
        SimpleNamespace(coordinate_type="unknown"), None,
    ) is None
    assert _plot_sigma_map(
        SimpleNamespace(median=SimpleNamespace(coordinate_type="unknown")), None,
    ) is None

    # Collapse to 0D returns None
    regbin_2d = SimpleNamespace(
        name="test", coordinate_type="regbin",
        coordinate_name=["x", "y"],
        coordinate_lowers=[0.0, 0.0], coordinate_uppers=[3.0, 3.0],
        map=np.ones((3, 3)),
    )
    assert _plot_regular_map(regbin_2d, {"x": None, "y": None}) is None

    sigma_2d = _make_mock_sigma(["x", "y"], [0.0, 0.0], [3.0, 3.0], (3, 3))
    assert _plot_sigma_map(sigma_2d, {"x": None, "y": None}) is None

    # SigmaMap routing for 1D and 2D regbin
    sigma_1d = _make_mock_sigma(["x"], [0.0], [3.0], (3,))
    fig, _ = _plot_sigma_map(sigma_1d, None)
    assert fig is not None

    fig, _ = _plot_sigma_map(sigma_2d, None)
    assert fig is not None
