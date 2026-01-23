"""Plugin-level profiler for appletree."""

import time
from typing import Dict, List, Optional, Any

import numpy as np

import appletree as apt
from appletree import randgen
from appletree.component import ComponentSim
from appletree.utils import exporter

export, __all__ = exporter()


@export
def profile_plugin(
    plugin_instance,
    key,
    parameters: Dict[str, float],
    input_data: Dict[str, Any],
    n_warmup: int = 2,
    n_runs: int = 10,
):
    """Profile a single plugin.

    Args:
        plugin_instance: Instantiated plugin object.
        key: Seed for JAX random generator.
        parameters: Dictionary of parameters.
        input_data: Dictionary mapping data names to arrays.
        n_warmup: Number of warmup runs for JIT compilation.
        n_runs: Number of timed runs.

    Returns:
        Tuple of (updated_key, output_data_dict, mean_time_ms, std_time_ms).

    """
    args = [input_data[dep] for dep in plugin_instance.depends_on]

    # Warmup runs for JIT compilation
    for _ in range(n_warmup):
        key, *outputs = plugin_instance.simulate(key, parameters, *args)
        for out in outputs:
            out.block_until_ready()

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        key, *outputs = plugin_instance.simulate(key, parameters, *args)
        for out in outputs:
            out.block_until_ready()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    output_data = {}
    for i, name in enumerate(plugin_instance.provides):
        output_data[name] = outputs[i]

    return key, output_data, np.mean(times), np.std(times)


@export
def profile_component(
    component: ComponentSim,
    parameters: Dict[str, float],
    batch_size: int = 1_000_000,
    n_warmup: int = 2,
    n_runs: int = 10,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Profile all plugins in a component.

    Args:
        component: A ComponentSim instance (must have been deduced).
        parameters: Dictionary of parameters.
        batch_size: Number of events to simulate.
        n_warmup: Number of warmup runs for JIT compilation.
        n_runs: Number of timed runs.
        verbose: Whether to print progress.

    Returns:
        List of dicts with profiling results for each plugin.

    """
    if not hasattr(component, "worksheet") or component.worksheet is None:
        raise RuntimeError(
            f"Component {component.name} has no worksheet. " "Make sure deduce() has been called."
        )

    results = []
    key = randgen.get_key()
    data_store: Dict[str, Any] = {"batch_size": batch_size}

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Profiling component: {component.name}")
        print(f"Likelihood: {component.llh_name}")
        print(f"Batch size: {batch_size:,}")
        print(f"Warmup runs: {n_warmup}, Timed runs: {n_runs}")
        print(f"{'=' * 60}")
        print(f"\n{'Plugin':<35} {'Time (ms)':<15} {'Std (ms)':<12} {'Provides'}")
        print("-" * 80)

    total_time = 0.0

    for work in component.worksheet:
        plugin_name = work[0]
        provides = work[1]

        plugin_class = component._plugin_class_registry[provides[0]]
        plugin_instance = plugin_class(component.llh_name)

        key, output_data, mean_time, std_time = profile_plugin(
            plugin_instance,
            key,
            parameters,
            data_store,
            n_warmup=n_warmup,
            n_runs=n_runs,
        )

        data_store.update(output_data)
        total_time += mean_time

        result = {
            "plugin": plugin_name,
            "provides": provides,
            "depends_on": work[2],
            "mean_time_ms": mean_time,
            "std_time_ms": std_time,
        }
        results.append(result)

        if verbose:
            provides_str = ", ".join(provides)
            print(f"{plugin_name:<35} {mean_time:>10.3f}     {std_time:>8.3f}     {provides_str}")

    if verbose:
        print("-" * 80)
        print(f"{'TOTAL':<35} {total_time:>10.3f} ms")
        print()

    return results


@export
def profile_context(
    context: "apt.Context",
    batch_size: int = 1_000_000,
    n_warmup: int = 2,
    n_runs: int = 10,
    parameters: Optional[Dict[str, float]] = None,
    verbose: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Profile all plugins in all components of a Context.

    Args:
        context: An appletree.Context instance.
        batch_size: Number of events to simulate.
        n_warmup: Number of warmup runs for JIT compilation.
        n_runs: Number of timed runs.
        parameters: Optional parameter dict. If None, uses sampled parameters.
        verbose: Whether to print progress.

    Returns:
        Dictionary mapping "likelihood_name/component_name" to profiling results.

    """
    if parameters is None:
        context.par_manager.sample_init()
        parameters = context.par_manager.get_all_parameter()

    if verbose:
        print("\n" + "=" * 70)
        print("APPLETREE PLUGIN PROFILER")
        print("=" * 70)
        print(f"\nParameters used: {len(parameters)} total")

    all_results = {}

    for llh_name, likelihood in context.likelihoods.items():
        if verbose:
            print(f"\n{'#' * 70}")
            print(f"# LIKELIHOOD: {llh_name}")
            print(f"{'#' * 70}")

        parameters_component = likelihood.replace_alias(parameters)
        for comp_name, component in likelihood.components.items():
            if not isinstance(component, ComponentSim):
                if verbose:
                    print(f"\nSkipping {comp_name} (not a simulation component)")
                continue

            # When the context is created, the components should already have a worksheet
            assert hasattr(component, "worksheet") and component.worksheet is not None

            results = profile_component(
                component,
                parameters_component,
                batch_size=batch_size,
                n_warmup=n_warmup,
                n_runs=n_runs,
                verbose=verbose,
            )

            key = f"{llh_name}/{comp_name}"
            all_results[key] = results

    if verbose:
        print_profile_summary(all_results)

    return all_results


@export
def print_profile_summary(all_results: Dict[str, List[Dict[str, Any]]], top_n: int = 5):
    """Print summary of profiling results.

    Args:
        all_results: Dictionary of profiling results from profile_context.
        top_n: Number of top slowest plugins to display.

    """
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    grand_total = 0.0
    for key, results in all_results.items():
        component_total = sum(r["mean_time_ms"] for r in results)
        grand_total += component_total
        print(f"{key}: {component_total:.3f} ms ({len(results)} plugins)")

    print("-" * 70)
    print(f"Grand total: {grand_total:.3f} ms")
    print()

    all_plugins = []
    for key, results in all_results.items():
        for r in results:
            all_plugins.append((r["plugin"], r["mean_time_ms"], key))

    all_plugins.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop {top_n} slowest plugins:")
    print("-" * 50)
    for i, (plugin, time_ms, comp) in enumerate(all_plugins[:top_n], 1):
        print(f"{i}. {plugin}: {time_ms:.3f} ms ({comp})")


@export
def profile_full_simulation(
    context: "apt.Context",
    batch_size: int = 1_000_000,
    n_warmup: int = 2,
    n_runs: int = 10,
    parameters: Optional[Dict[str, float]] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Profile the full simulation pipeline (not individual plugins).

    Args:
        context: An appletree.Context instance.
        batch_size: Number of events to simulate.
        n_warmup: Number of warmup runs for JIT compilation.
        n_runs: Number of timed runs.
        parameters: Optional parameter dict. If None, uses sampled parameters.
        verbose: Whether to print progress.

    Returns:
        Dictionary mapping component names to timing results.

    """
    if parameters is None:
        context.par_manager.sample_init()
        parameters = context.par_manager.get_all_parameter()

    results = {}

    for llh_name, likelihood in context.likelihoods.items():
        parameters_component = likelihood.replace_alias(parameters)
        for comp_name, component in likelihood.components.items():
            if not isinstance(component, ComponentSim):
                if verbose:
                    print(f"\nSkipping {comp_name} (not a simulation component)")
                continue

            # When the context is created, the components should already have a simulate method
            assert hasattr(component, "simulate") and component.simulate is not None

            key = randgen.get_key()

            # Warmup
            for _ in range(n_warmup):
                key, result = component.simulate(key, batch_size, parameters_component)
                for r in result:
                    r.block_until_ready()

            # Timed runs
            times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                key, result = component.simulate(key, batch_size, parameters_component)
                for r in result:
                    r.block_until_ready()
                end = time.perf_counter()
                times.append((end - start) * 1000)

            comp_key = f"{llh_name}/{comp_name}"
            results[comp_key] = {
                "mean_time_ms": np.mean(times),
                "std_time_ms": np.std(times),
            }

            if verbose:
                print(
                    f"{comp_key}: {np.mean(times):.3f} +/- {np.std(times):.3f} ms "
                    f"(full pipeline)"
                )

    return results


@export
def compare_plugin_vs_full(
    context: "apt.Context",
    batch_size: int = 1_000_000,
    n_warmup: int = 2,
    n_runs: int = 10,
    parameters: Optional[Dict[str, float]] = None,
):
    """Compare individual plugin timing vs full pipeline timing.

    Args:
        context: An appletree.Context instance.
        batch_size: Number of events to simulate.
        n_warmup: Number of warmup runs for JIT compilation.
        n_runs: Number of timed runs.
        parameters: Optional parameter dict.

    """
    if parameters is None:
        context.par_manager.sample_init()
        parameters = context.par_manager.get_all_parameter()

    print("\n" + "=" * 70)
    print("COMPARING INDIVIDUAL PLUGINS VS FULL PIPELINE")
    print("=" * 70)

    plugin_results = profile_context(
        context, batch_size, n_warmup, n_runs, parameters, verbose=False
    )

    full_results = profile_full_simulation(
        context, batch_size, n_warmup, n_runs, parameters, verbose=False
    )

    print(f"\n{'Component':<40} {'Plugins (ms)':<15} {'Full (ms)':<15} {'Overhead'}")
    print("-" * 80)

    for comp_key in plugin_results:
        plugin_total = sum(r["mean_time_ms"] for r in plugin_results[comp_key])
        full_time = full_results[comp_key]["mean_time_ms"]
        overhead = (plugin_total / full_time - 1) * 100 if full_time > 0 else 0

        print(
            f"{comp_key:<40} {plugin_total:>10.3f}     {full_time:>10.3f}     "
            f"{overhead:>+6.1f}%"
        )

    print()
    print("Note: Positive overhead means individual plugins are slower than full pipeline")
    print("      (expected due to JIT fusion in full pipeline)")


@export
def print_component_code(component: ComponentSim):
    """Print the generated code for a component.

    Args:
        component: A ComponentSim instance (must have been deduced).

    """
    if not hasattr(component, "code") or component.code is None:
        print("Component has no generated code. Run deduce() first.")
        return

    print(f"\nGenerated code for component: {component.name}")
    print("=" * 60)
    print(component.code)
    print("=" * 60)


@export
def print_worksheet(component: ComponentSim):
    """Print the worksheet (execution order) for a component.

    Args:
        component: A ComponentSim instance (must have been deduced).

    """
    if not hasattr(component, "worksheet") or component.worksheet is None:
        print("Component has no worksheet. Run deduce() first.")
        return

    print(f"\nWorksheet for component: {component.name}")
    print("=" * 60)
    print(f"{'#':<4} {'Plugin':<30} {'Provides':<25} {'Depends On'}")
    print("-" * 90)

    for i, work in enumerate(component.worksheet, 1):
        plugin = work[0]
        provides = ", ".join(work[1])
        depends_on = ", ".join(work[2]) if work[2] else "(none)"
        print(f"{i:<4} {plugin:<30} {provides:<25} {depends_on}")

    print()
