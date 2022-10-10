import os
import pandas as pd
import appletree as apt
from jax import numpy as jnp

# Get parameters
par_config_file_name = os.path.join(apt.PARPATH, 'apt_er_sr0.json')
par_manager = apt.Parameter(par_config_file_name)
par_manager.sample_init()
parameters = par_manager.get_all_parameter()

# Define bins
data_file_name = os.path.join(
    apt.DATAPATH,
    'data_Rn220.csv',
)
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
    )
    ac.deduce(data_names = ('cs1', 'cs2'))
    ac.simulate_hist(parameters)


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
    er.rate_name = 'er_rate'
    key = apt.randgen.get_key(seed=137)
    er.simulate_hist(key, int(1e3), parameters)
