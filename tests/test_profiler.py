import pytest

import appletree as apt
from appletree.share import _cached_configs, _cached_functions
from appletree.component import ComponentSim


@pytest.fixture(scope="module")
def rn220_context():
    """Create a shared Rn220 context for all tests in this module."""
    _cached_functions.clear()
    _cached_configs.clear()
    context = apt.ContextRn220()
    context.par_manager.sample_init()
    return context


def test_profile_context(rn220_context):
    """Test profile_context function."""
    results = apt.profile_context(
        rn220_context,
        batch_size=int(1e3),
        n_warmup=1,
        n_runs=2,
        verbose=False,
    )

    # Check structure of results
    assert isinstance(results, dict)
    assert len(results) > 0

    for key, plugin_results in results.items():
        assert isinstance(plugin_results, list)
        assert len(plugin_results) > 0

        for r in plugin_results:
            assert "plugin" in r
            assert "provides" in r
            assert "depends_on" in r
            assert "mean_time_ms" in r
            assert "std_time_ms" in r
            assert r["mean_time_ms"] >= 0
            assert r["std_time_ms"] >= 0


def test_profile_context_verbose(rn220_context, capsys):
    """Test profile_context with verbose=True."""
    results = apt.profile_context(
        rn220_context,
        batch_size=int(1e3),
        n_warmup=1,
        n_runs=2,
        verbose=True,
    )

    captured = capsys.readouterr()
    # Check verbose output
    assert "APPLETREE PLUGIN PROFILER" in captured.out
    assert "LIKELIHOOD" in captured.out
    assert "SUMMARY" in captured.out
    assert "Top 5 slowest plugins" in captured.out
    assert "Grand total" in captured.out


def test_profile_component(rn220_context):
    """Test profile_component function."""
    parameters = rn220_context.par_manager.get_all_parameter()
    component = rn220_context.likelihoods["rn220_llh"].components["rn220_er"]
    assert isinstance(component, ComponentSim)

    results = apt.profile_component(
        component,
        parameters,
        batch_size=int(1e3),
        n_warmup=1,
        n_runs=2,
        verbose=False,
    )

    assert isinstance(results, list)
    assert len(results) == len(component.worksheet)

    for r in results:
        assert r["mean_time_ms"] >= 0


def test_profile_component_verbose(rn220_context, capsys):
    """Test profile_component with verbose=True."""
    parameters = rn220_context.par_manager.get_all_parameter()
    component = rn220_context.likelihoods["rn220_llh"].components["rn220_er"]

    results = apt.profile_component(
        component,
        parameters,
        batch_size=int(1e3),
        n_warmup=1,
        n_runs=2,
        verbose=True,
    )

    captured = capsys.readouterr()
    assert "Profiling component" in captured.out
    assert "Batch size" in captured.out
    assert "TOTAL" in captured.out


def test_profile_full_simulation(rn220_context):
    """Test profile_full_simulation function."""
    results = apt.profile_full_simulation(
        rn220_context,
        batch_size=int(1e3),
        n_warmup=1,
        n_runs=2,
        verbose=False,
    )

    assert isinstance(results, dict)
    assert len(results) > 0

    for key, timing in results.items():
        assert "mean_time_ms" in timing
        assert "std_time_ms" in timing
        assert timing["mean_time_ms"] >= 0


def test_profile_full_simulation_verbose(rn220_context, capsys):
    """Test profile_full_simulation with verbose=True."""
    results = apt.profile_full_simulation(
        rn220_context,
        batch_size=int(1e3),
        n_warmup=1,
        n_runs=2,
        verbose=True,
    )

    captured = capsys.readouterr()
    assert "full pipeline" in captured.out


def test_compare_plugin_vs_full(rn220_context, capsys):
    """Test compare_plugin_vs_full function."""
    apt.compare_plugin_vs_full(
        rn220_context,
        batch_size=int(1e3),
        n_warmup=1,
        n_runs=2,
    )

    captured = capsys.readouterr()
    assert "COMPARING INDIVIDUAL PLUGINS VS FULL PIPELINE" in captured.out
    assert "Overhead" in captured.out


def test_print_functions(rn220_context, capsys):
    """Test print_worksheet and print_component_code functions."""
    component = rn220_context.likelihoods["rn220_llh"].components["rn220_er"]

    # Test print_worksheet
    apt.profiler.print_worksheet(component)
    captured = capsys.readouterr()
    assert "Worksheet for component" in captured.out
    assert "Plugin" in captured.out
    assert "Provides" in captured.out

    # Test print_component_code
    apt.profiler.print_component_code(component)
    captured = capsys.readouterr()
    assert "Generated code for component" in captured.out
    assert "@partial(jit" in captured.out


def test_print_functions_no_worksheet(capsys):
    """Test print functions when component has no worksheet."""
    _cached_functions.clear()

    # Create a component without calling deduce()
    component = apt.components.ERBand(
        name="test_component",
        llh_name="test_llh",
    )

    # Test print_worksheet with no worksheet
    apt.profiler.print_worksheet(component)
    captured = capsys.readouterr()
    assert "Component has no worksheet" in captured.out

    # Test print_component_code with no code
    apt.profiler.print_component_code(component)
    captured = capsys.readouterr()
    assert "Component has no generated code" in captured.out
