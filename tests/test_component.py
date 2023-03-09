import pandas as pd
from jax import numpy as jnp

import appletree as apt
from appletree.utils import get_file_path


# Get parameters
par_config_file_name = get_file_path('er.json')
par_manager = apt.Parameter(par_config_file_name)
par_manager.sample_init()
parameters = par_manager.get_all_parameter()

# Define bins
data_file_name = get_file_path('data_Rn220.csv',)
data = pd.read_csv(data_file_name)
data = data[['cs1', 'cs2']].to_numpy()
bins_cs1, bins_cs2 = apt.utils.get_equiprob_bins_2d(
    data,
    [15, 15],
    order = [0, 1],
    x_clip = [0, 100],
    y_clip = [1e2, 1e4],
    which_np = jnp,
)


def test_fixed_component():
    """Test ComponentFixed"""
    ac = apt.components.AC(
        bins = [bins_cs1, bins_cs2],
        bins_type = 'irreg',
        file_name = 'AC_Rn220.pkl',
    )
    ac.rate_name = 'ac_rate'
    ac.deduce(data_names = ('cs1', 'cs2'))
    ac.simulate_hist(parameters)
    ac.simulate_weighed_data(parameters)


def test_sim_component():
    """Test ComponentSim"""
    er = apt.components.ERBand(
        bins = [bins_cs1, bins_cs2],
        bins_type = 'irreg',
    )
    er.deduce(
        data_names = ('cs1', 'cs2'),
        func_name = 'er_sim',
    )
    er.compile()
    er.save_code('_temp.json')
    er.rate_name = 'er_rate'
    batch_size = int(1e3)
    key = apt.randgen.get_key(seed=137)
    er.simulate_hist(key, batch_size, parameters)
    er.simulate_weighed_data(key, batch_size, parameters)

    @apt.utils.timeit
    def test(key, batch_size, parameters):
        return er.simulate_hist(key, batch_size, parameters)

    @apt.utils.timeit
    def benchmark(key):
        for _ in range(100):
            key, _ = test(key, batch_size, parameters)

    benchmark(key)
