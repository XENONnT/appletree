import os
import pandas as pd
import appletree as apt
from jax import numpy as jnp

# Get parameters
par_config_file_name = os.path.join(apt.PARPATH, 'apt_sr0_er.json')
par_manager = apt.Parameter(par_config_file_name)

par_manager.sample_init()
parameters = par_manager.get_all_parameter()

# Define bins
data_file_name = os.path.join(apt.DATAPATH, 'data_XENONnT_Rn220_v8_strax_v1.2.2_straxen_v1.7.1_cutax_v1.9.0.csv')
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

# Fixed component
ac = apt.components.AC(
    bins = [bins_cs1, bins_cs2],
    bins_type = 'irreg',
)
ac.deduce(data_names = ('cs1', 'cs2'))
ac_hist = ac.simulate_hist(parameters)

# Simulation component
er = apt.components.ERBand(
    bins = [bins_cs1, bins_cs2],
    bins_type='irreg',
)
er.deduce(
    data_names = ('cs1', 'cs2'), 
    func_name = 'er_sim',
)
er.compile()
er.rate_name = 'er_rate'
key = apt.randgen.get_key(seed=137)
key, er_hist = er.simulate_hist(key, int(1e6), parameters)
