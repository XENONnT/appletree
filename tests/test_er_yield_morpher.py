import json

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

import appletree as apt
from appletree.plugins.er_nestv2 import ERYieldMorpher, ExcitonIonRatioER, LyER, QyER
from appletree.share import _cached_configs, _cached_functions, set_global_config
from appletree.utils import load_json


@pytest.fixture(autouse=True)
def clear_appletree_caches():
    """Keep map and compiled-function globals isolated between tests."""
    _cached_configs.clear()
    _cached_functions.clear()
    yield
    _cached_configs.clear()
    _cached_functions.clear()


def _write_uncertainty_map(tmp_path, values, coordinates=(0.0, 1.0, 2.0)):
    """Write an artificial energy-dependent uncertainty map for a test."""
    path = tmp_path / "er_ly_rel_uncertainty.json"
    path.write_text(
        json.dumps(
            {
                "coordinate_type": "point",
                "coordinate_name": "energy",
                "coordinate_system": list(coordinates),
                "map": list(values),
            }
        )
    )
    return str(path)


def _nominal_parameters():
    """Build deterministic nominal parameters from the checked-in ER instruction."""
    instruction = load_json("er_nestv2.json")
    parameters = {}
    for name, setting in instruction.items():
        prior_args = setting.get("prior_args", {})
        parameters[name] = prior_args.get("val", setting.get("init_mean"))
    parameters["t_er_yield"] = 0.0
    return parameters


def _nominal_qy(energies, parameters=None):
    parameters = _nominal_parameters() if parameters is None else parameters
    energy = jnp.asarray(energies)
    key = apt.randgen.get_key(seed=11)
    key, nex_ni_ratio, _ = ExcitonIonRatioER().simulate(key, parameters, energy)
    _, charge_yield = QyER().simulate(
        key,
        parameters,
        energy,
        nex_ni_ratio,
        parameters["field"],
    )
    return np.asarray(charge_yield)


def _simulate_morpher(tmp_path, map_values, energies, charge_yield, t_er_yield, w=0.01):
    map_path = _write_uncertainty_map(tmp_path, map_values)
    set_global_config({"er_ly_rel_uncertainty": map_path})
    morpher = ERYieldMorpher(llh_name="er_yield_morpher_test")
    parameters = {"w": w, "t_er_yield": t_er_yield}
    key = apt.randgen.get_key(seed=19)
    _, morphed = morpher.simulate(
        key,
        parameters,
        jnp.asarray(energies),
        jnp.asarray(charge_yield),
    )
    return morpher, np.asarray(morphed)


def test_nominal_qyer_is_unchanged_and_lyer_keeps_old_formula(tmp_path):
    energies = np.asarray([2.0, 5.0, 20.0])
    parameters = _nominal_parameters()
    charge_yield = _nominal_qy(energies, parameters)
    assert np.all(charge_yield < 1.0 / parameters["w"])

    _, morphed = _simulate_morpher(
        tmp_path,
        [0.2, 0.2, 0.2],
        energies,
        charge_yield,
        t_er_yield=0.0,
        w=parameters["w"],
    )
    npt.assert_allclose(morphed, charge_yield, rtol=1e-5, atol=1e-5)

    key = apt.randgen.get_key(seed=23)
    _, light_yield = LyER().simulate(
        key,
        parameters,
        jnp.asarray(morphed),
    )
    expected = np.maximum(1.0 / parameters["w"] - charge_yield, 0.0)
    npt.assert_allclose(np.asarray(light_yield), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("t_er_yield", [-1.0, -0.4, 0.0, 0.6, 1.0])
def test_morphed_light_and_charge_yields_conserve_quanta(tmp_path, t_er_yield):
    energies = np.asarray([-1.0, 0.0, 0.25, 1.0, 1.75, 2.0, 3.0])
    charge_yield = np.asarray([5.0, 15.0, 35.0, 50.0, 65.0, 80.0, 95.0])
    w = 0.01
    _, morphed = _simulate_morpher(
        tmp_path,
        [0.05, 0.2, 0.35],
        energies,
        charge_yield,
        t_er_yield,
        w=w,
    )
    key = apt.randgen.get_key(seed=29)
    _, light_yield = LyER().simulate(key, {"w": w}, jnp.asarray(morphed))
    npt.assert_allclose(morphed + np.asarray(light_yield), 1.0 / w, rtol=1e-5, atol=1e-5)


def test_plus_minus_one_scale_light_yield_and_complement_charge_yield(tmp_path):
    energies = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0])
    charge_yield = np.asarray([20.0, 40.0, 60.0, 70.0, 80.0])
    w = 0.01

    _, nominal = _simulate_morpher(
        tmp_path,
        [0.2, 0.2, 0.2],
        energies,
        charge_yield,
        t_er_yield=0.0,
        w=w,
    )
    _, plus = _simulate_morpher(
        tmp_path,
        [0.2, 0.2, 0.2],
        energies,
        charge_yield,
        t_er_yield=1.0,
        w=w,
    )
    _, minus = _simulate_morpher(
        tmp_path,
        [0.2, 0.2, 0.2],
        energies,
        charge_yield,
        t_er_yield=-1.0,
        w=w,
    )

    light_nominal = 1.0 / w - nominal
    npt.assert_allclose(1.0 / w - plus, 1.2 * light_nominal, rtol=1e-5, atol=1e-5)
    npt.assert_allclose(1.0 / w - minus, 0.8 * light_nominal, rtol=1e-5, atol=1e-5)
    npt.assert_allclose(plus - charge_yield, -0.2 * light_nominal, rtol=1e-5, atol=1e-5)
    npt.assert_allclose(minus - charge_yield, 0.2 * light_nominal, rtol=1e-5, atol=1e-5)
    assert not np.allclose(plus, 1.2 * charge_yield)
    assert not np.allclose(minus, 0.8 * charge_yield)
    assert not np.allclose(plus, 0.8 * charge_yield)
    assert not np.allclose(minus, 1.2 * charge_yield)


def test_lerp_map_interpolates_endpoints_and_holds_outside(tmp_path):
    energies = np.asarray([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    charge_yield = np.full(len(energies), 40.0)
    morpher, morphed = _simulate_morpher(
        tmp_path,
        [0.1, 0.2, 0.4],
        energies,
        charge_yield,
        t_er_yield=1.0,
        w=0.01,
    )
    uncertainty = np.asarray(morpher.er_ly_rel_uncertainty.apply(jnp.asarray(energies)))
    expected_uncertainty = np.interp(energies, [0.0, 1.0, 2.0], [0.1, 0.2, 0.4])
    npt.assert_allclose(uncertainty, expected_uncertainty, rtol=1e-5, atol=1e-5)

    expected_light = np.clip((100.0 - charge_yield) * (1.0 + expected_uncertainty), 0.0, 100.0)
    npt.assert_allclose(morphed, 100.0 - expected_light, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("t_er_yield", [-100.0, -5.0, 0.0, 5.0, 100.0])
def test_extreme_parameter_boundaries_stay_in_quanta_bounds(tmp_path, t_er_yield):
    energies = np.asarray([0.0, 1.0, 2.0])
    charge_yield = np.asarray([0.0, 50.0, 80.0])
    _, morphed = _simulate_morpher(
        tmp_path,
        [0.5, 0.5, 0.5],
        energies,
        charge_yield,
        t_er_yield,
        w=0.01,
    )
    key = apt.randgen.get_key(seed=31)
    _, light_yield = LyER().simulate(key, {"w": 0.01}, jnp.asarray(morphed))
    light_yield = np.asarray(light_yield)
    assert np.all((morphed >= 0.0) & (morphed <= 100.0))
    assert np.all((light_yield >= 0.0) & (light_yield <= 100.0))
    npt.assert_allclose(morphed + light_yield, 100.0, rtol=1e-5, atol=1e-5)
    expected_light = np.clip((100.0 - charge_yield) * (1.0 + 0.5 * t_er_yield), 0.0, 100.0)
    npt.assert_allclose(light_yield, expected_light, rtol=1e-5, atol=1e-5)
    if t_er_yield <= -5.0:
        npt.assert_allclose(light_yield, 0.0, rtol=1e-5, atol=1e-5)
    if t_er_yield >= 100.0:
        npt.assert_allclose(light_yield, 100.0, rtol=1e-5, atol=1e-5)


def test_default_zero_uncertainty_has_no_effect_on_physical_yields():
    energies = np.asarray([2.0, 5.0, 20.0])
    parameters = _nominal_parameters()
    charge_yield = _nominal_qy(energies, parameters)
    assert np.all(charge_yield < 1.0 / parameters["w"])
    morpher = ERYieldMorpher(llh_name="er_yield_morpher_default_test")
    parameters["t_er_yield"] = 1.0
    key = apt.randgen.get_key(seed=37)
    _, morphed = morpher.simulate(
        key,
        parameters,
        jnp.asarray(energies),
        jnp.asarray(charge_yield),
    )
    npt.assert_allclose(np.asarray(morphed), charge_yield, rtol=1e-5, atol=1e-5)


def test_t_zero_clips_unphysical_charge_yield_to_total_quanta(tmp_path):
    energies = np.asarray([0.0, 1.0])
    total_quanta = 100.0
    charge_yield = np.asarray([101.0, 130.0])
    _, morphed = _simulate_morpher(
        tmp_path,
        [0.2, 0.2, 0.2],
        energies,
        charge_yield,
        t_er_yield=0.0,
        w=0.01,
    )
    npt.assert_allclose(morphed, total_quanta, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "component_class", [apt.components.ERBandNestV2, apt.components.ERPeakNestV2]
)
def test_compiled_nestv2_graph_propagates_morphed_yields(tmp_path, component_class):
    map_path = _write_uncertainty_map(tmp_path, [0.2, 0.2, 0.2])
    set_global_config({"er_ly_rel_uncertainty": map_path})
    name = component_class.__name__
    component = component_class(name=name + "_morpher_test", llh_name=name + "_morpher_llh")
    output_names = [
        "energy",
        "charge_yield",
        "charge_yield_morphed",
        "light_yield",
        "_Nph",
        "_Ne",
    ]
    component.deduce(
        data_names=output_names,
        func_name=name + "_morpher_sim",
        force_no_eff=True,
    )
    assert any(row[0] == "ERYieldMorpher" for row in component.worksheet)
    component.compile()

    parameters = _nominal_parameters()
    parameters["t_er_yield"] = 0.25
    key = apt.randgen.get_key(seed=41)
    _, result = component.simulate(key, 16, parameters)
    energy, charge_yield, morphed, light_yield, nph, ne = [np.asarray(r) for r in result]

    total_quanta = 1.0 / parameters["w"]
    npt.assert_allclose(morphed + light_yield, total_quanta, rtol=1e-5, atol=1e-5)
    npt.assert_allclose(nph, light_yield * energy, rtol=1e-5, atol=1e-5)
    npt.assert_allclose(ne, morphed * energy, rtol=1e-5, atol=1e-5)
    assert np.any(np.abs(morphed - charge_yield) > 1e-6)
