"""Tests for mixed per-axis coordinate_type in Map."""

import sys
import json
import numpy as np
import numpy.testing as npt
from jax import numpy as jnp

# Import config directly to avoid full appletree init which pulls in aptext
from appletree.share import _cached_configs
from appletree.config import Map


def _build_map_from_dict(data, name="test_map"):
    """Build a Map from a dict (bypassing file I/O)."""
    _cached_configs.clear()
    m = Map(name=name, default="dummy")
    m.file_path = name
    if isinstance(data["coordinate_type"], list):
        m.build_regbin(data)
    elif data["coordinate_type"] in ("regbin", "log_regbin"):
        m.build_regbin(data)
    return m


def test_mixed_coordinate_type_2d():
    """A 2D map with linear axis 0 and log axis 1."""
    # 3x4 map: axis 0 is quantile [0, 1], axis 1 is S2 area [100, 10000]
    map_vals = np.arange(12, dtype=float).reshape(3, 4)
    data = {
        "coordinate_type": ["regbin", "log_regbin"],
        "coordinate_name": ["quantile", "s2_area"],
        "coordinate_lowers": [0.0, 100.0],
        "coordinate_uppers": [1.0, 10000.0],
        "map": map_vals.tolist(),
    }
    m = _build_map_from_dict(data)

    assert m._is_log_axis == [False, True]
    assert isinstance(m.coordinate_type, list)

    # Test at grid nodes: should return exact map values
    # Grid node (0, 0): quantile=0.0, s2_area=100.0
    pos = jnp.array([[0.0, 100.0]])
    val = m.apply(pos)
    npt.assert_allclose(val, [map_vals[0, 0]], atol=1e-5)

    # Grid node (2, 3): quantile=1.0, s2_area=10000.0
    pos = jnp.array([[1.0, 10000.0]])
    val = m.apply(pos)
    npt.assert_allclose(val, [map_vals[2, 3]], atol=1e-5)


def test_mixed_vs_uniform_regbin():
    """Uniform regbin should give same results as mixed with all-linear."""
    map_vals = np.random.default_rng(42).random((5, 6))
    data_uniform = {
        "coordinate_type": "regbin",
        "coordinate_name": ["x", "y"],
        "coordinate_lowers": [0.0, 0.0],
        "coordinate_uppers": [4.0, 5.0],
        "map": map_vals.tolist(),
    }
    data_mixed = {
        "coordinate_type": ["regbin", "regbin"],
        "coordinate_name": ["x", "y"],
        "coordinate_lowers": [0.0, 0.0],
        "coordinate_uppers": [4.0, 5.0],
        "map": map_vals.tolist(),
    }
    m_uniform = _build_map_from_dict(data_uniform, name="uniform")
    m_mixed = _build_map_from_dict(data_mixed, name="mixed")

    pos = jnp.array([[1.0, 2.0], [2.5, 3.5], [0.1, 4.9]])
    npt.assert_allclose(m_uniform.apply(pos), m_mixed.apply(pos), atol=1e-6)


def test_mixed_vs_uniform_log_regbin():
    """Uniform log_regbin should give same results as mixed with all-log."""
    map_vals = np.random.default_rng(42).random((5, 6))
    data_uniform = {
        "coordinate_type": "log_regbin",
        "coordinate_name": ["x", "y"],
        "coordinate_lowers": [1.0, 10.0],
        "coordinate_uppers": [1000.0, 100000.0],
        "map": map_vals.tolist(),
    }
    data_mixed = {
        "coordinate_type": ["log_regbin", "log_regbin"],
        "coordinate_name": ["x", "y"],
        "coordinate_lowers": [1.0, 10.0],
        "coordinate_uppers": [1000.0, 100000.0],
        "map": map_vals.tolist(),
    }
    m_uniform = _build_map_from_dict(data_uniform, name="uniform_log")
    m_mixed = _build_map_from_dict(data_mixed, name="mixed_log")

    pos = jnp.array([[10.0, 100.0], [100.0, 1000.0], [500.0, 50000.0]])
    npt.assert_allclose(m_uniform.apply(pos), m_mixed.apply(pos), atol=1e-6)


def test_mixed_coordinate_type_length_mismatch():
    """coordinate_type list length must match number of axes."""
    data = {
        "coordinate_type": ["regbin"],
        "coordinate_name": ["x", "y"],
        "coordinate_lowers": [0.0, 0.0],
        "coordinate_uppers": [1.0, 1.0],
        "map": [[1.0, 2.0], [3.0, 4.0]],
    }
    try:
        _build_map_from_dict(data)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must match" in str(e)


def test_mixed_coordinate_log_pdf_validation():
    """Log-scaled pdf/cdf axis should raise an error."""
    data = {
        "coordinate_type": ["log_regbin", "regbin"],
        "coordinate_name": ["pdf", "x"],
        "coordinate_lowers": [0.01, 0.0],
        "coordinate_uppers": [1.0, 1.0],
        "map": [[1.0, 2.0], [3.0, 4.0]],
    }
    try:
        _build_map_from_dict(data)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # pdf on a linear axis with log on the other axis should be fine
    data["coordinate_type"] = ["regbin", "log_regbin"]
    data["coordinate_name"] = ["pdf", "x"]
    data["coordinate_lowers"] = [0.01, 1.0]
    data["coordinate_uppers"] = [1.0, 100.0]
    m = _build_map_from_dict(data)
    assert m._is_log_axis == [False, True]
