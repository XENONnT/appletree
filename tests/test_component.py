import pytest

import numpy as np
import pandas as pd

import appletree as apt
from appletree.utils import get_file_path
from appletree.share import _cached_functions


# Get parameters
par_instruct_file_name = get_file_path("er.json")
par_manager = apt.Parameter(par_instruct_file_name)
par_manager.sample_init()
parameters = par_manager.get_all_parameter()

# Define bins
data_file_name = get_file_path(
    "data_Rn220.csv",
)
data = pd.read_csv(data_file_name)
data = data[["cs1", "cs2"]].to_numpy()
bins_cs1, bins_cs2 = apt.utils.get_equiprob_bins_2d(
    data,
    [15, 15],
    order=[0, 1],
    x_clip=[0, 100],
    y_clip=[1e2, 1e4],
    which_np=np,
)


def test_fixed_component():
    """Test ComponentFixed."""
    ac = apt.components.AC(
        bins=[bins_cs1, bins_cs2],
        bins_type="irreg",
        file_name="AC_Rn220.pkl",
    )
    ac.rate_name = "ac_rate"
    ac.deduce(data_names=["cs1", "cs2"])
    ac.lineage_hash
    ac.simulate_hist(parameters)
    ac.simulate_weighted_data(parameters)


def test_sim_component():
    """Test ComponentSim."""
    _cached_functions.clear()
    er = apt.components.ERBand(
        bins=[bins_cs1, bins_cs2],
        bins_type="irreg",
    )
    er.deduce(
        data_names=["cs1", "cs2"],
        func_name="er_sim",
    )
    er.compile()
    er.save_code("_temp.json")
    er.rate_name = "er_rate"
    batch_size = int(1e3)
    key = apt.randgen.get_key(seed=137)

    key, r = er.multiple_simulations(key, batch_size, parameters, 5, apply_eff=True)

    key, h = er.simulate_hist(key, batch_size, parameters)
    apt.utils.plot_irreg_histogram_2d(*er.bins, h, density=False)

    er.simulate_weighted_data(key, batch_size, parameters)

    @apt.utils.timeit
    def test(key, batch_size, parameters):
        return er.simulate_hist(key, batch_size, parameters)

    @apt.utils.timeit
    def benchmark(key):
        for _ in range(100):
            key, _ = test(key, batch_size, parameters)

    benchmark(key)

    # if _cached_functions not cleared, this will raise an error
    with pytest.raises(RuntimeError):
        er.deduce(
            data_names=("cs1", "cs2"),
            func_name="er_sim",
            force_no_eff=True,
        )

    _cached_functions.clear()
    # re-deduce after clearing _cached_functions
    er.deduce(
        data_names=("cs1", "cs2"),
        func_name="er_sim",
        force_no_eff=True,
    )
    er.compile()
    er.lineage_hash
    er.simulate_hist(key, batch_size, parameters)
    with pytest.raises(RuntimeError):
        key, r = er.multiple_simulations(key, batch_size, parameters, 5, apply_eff=True)

    # test for multiple nodep_data_names
    with pytest.raises(RuntimeError):
        er.deduce(
            data_names=("cs1", "cs2"),
            func_name="er_sim",
            nodep_data_names=["x", "y", "z", "num_photon", "num_electron"],
            force_no_eff=True,
        )
    _cached_functions.clear()
    er.deduce(
        data_names=("cs1", "cs2"),
        func_name="er_sim",
        nodep_data_names=["x", "y", "z", "num_photon", "num_electron"],
        force_no_eff=True,
    )
    er.compile()
    er.lineage_hash
    x, y, z = np.zeros((3, batch_size))
    num_photon = np.random.randint(1, 100)
    num_electron = np.random.randint(1, 100)
    er.simulate(key, x, y, z, num_photon, num_electron, parameters)

    _cached_functions.clear()
    er.deduce(
        data_names=("cs1", "cs2"),
        func_name="er_sim",
        nodep_data_names=["batch_size", "num_photon", "num_electron"],
        force_no_eff=True,
    )
    er.compile()
    er.simulate(key, batch_size, num_photon, num_electron, parameters)
