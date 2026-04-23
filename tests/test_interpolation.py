"""Tests for every interpolator registered on ``Map`` — ``_POINT_INTERPOLATORS`` (1-D point-based)
and ``_REGBIN_INTERPOLATORS`` (2-D and 3-D regular-binning).

We sample the known function ``f(x) = sum(x_i)`` on a uniform grid in
``[0, 1]^ndim`` and check that each method returns the analytically-correct
value at a method-appropriate query point.

"""

import numpy as np
import numpy.testing as npt
import pytest
import jax.numpy as jnp

from appletree.share import _cached_configs
from appletree.config import Map

# Auto-discover every registered combination so the test grows with the table.
COMBOS = [(1, m) for m in Map._POINT_INTERPOLATORS] + list(Map._REGBIN_INTERPOLATORS.keys())


def _build_map(ndim, method, n_per_axis=5):
    """Build an in-memory map of ``f(x) = sum(x_i)`` on ``[0, 1]^ndim``."""
    _cached_configs.clear()
    axes = [np.linspace(0.0, 1.0, n_per_axis) for _ in range(ndim)]
    if ndim == 1:
        map_vals = axes[0]
        data = {
            "coordinate_type": "point",
            "coordinate_name": "x",
            "coordinate_system": axes[0].tolist(),
            "map": map_vals.tolist(),
        }
        builder = "build_point"
    else:
        map_vals = sum(np.meshgrid(*axes, indexing="ij"))
        data = {
            "coordinate_type": "regbin",
            "coordinate_name": [f"x{i}" for i in range(ndim)],
            "coordinate_lowers": [0.0] * ndim,
            "coordinate_uppers": [1.0] * ndim,
            "map": map_vals.tolist(),
        }
        builder = "build_regbin"

    m = Map(name=f"test_{ndim}d_{method}", method=method, default="dummy")
    m.file_path = m.name
    getattr(m, builder)(data)
    return m


@pytest.mark.parametrize("ndim,method", COMBOS)
def test_known_function(ndim, method):
    """Check each interpolator on the known function ``f(x) = sum(x_i)``.

    For each method we pick a query point whose answer is analytically
    determined:

      * **NN** at ``(0.4,) * ndim`` — the nearest grid corner is
        ``(0.5,) * ndim`` (grid step = 0.25), so NN must return
        ``f((0.5,) * ndim) = 0.5 * ndim``.
      * **IDW / LERP** at the cell center ``(0.375,) * ndim`` — all
        ``2^ndim`` corners of the surrounding cell are equidistant, so both
        methods reduce to the corner average. For a linear ``f`` the corner
        average equals ``f(center) = 0.375 * ndim``.

    """
    m = _build_map(ndim, method, n_per_axis=5)

    if method == "NN":
        pos = np.full(ndim, 0.4, dtype=np.float32)
        expected = 0.5 * ndim
    else:
        pos = np.full(ndim, 0.375, dtype=np.float32)
        expected = 0.375 * ndim

    query = pos if ndim == 1 else pos[np.newaxis, :]
    out = np.asarray(m.apply(jnp.asarray(query)))
    npt.assert_allclose(out.ravel(), expected, rtol=1e-4, atol=1e-4)
