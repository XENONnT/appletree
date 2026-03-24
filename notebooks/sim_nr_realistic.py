# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: XENONnT_el7.2025.03.1
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Simulating some NR events
#
# Pueh Leng Tan, 10 March 2026

# %%
import os
import sys
from time import time

import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import jax.numpy as jnp
import multihist as mh
import json

import appletree as apt
from appletree.utils import get_file_path
from appletree.share import _cached_functions

import aptext

# %%
# constrain the GPU memory usage
apt.set_gpu_memory_usage(0.2)

# %%
#'''
MC_ID = int(sys.argv[1])
num_sims = int(sys.argv[2])
#'''

'''
MC_ID = 1

#num_sims = int(3000)
#num_sims = int(50)
num_sims = int(10)
#num_sims = int(1e4)
#'''

# %% [markdown]
# ## Define component

# %% [markdown]
# ### ComponentSim

# %%
# Initialize component
#nr = apt.NR(bins=[bins_cs1, bins_cs2], bins_type="irreg")
#nr = apt.NR()
#nr = apt.components.NRBand2f()

# https://github.com/XENONnT/applefiles/blob/cbb3139150526e647d1e2985f2079577f8218cd3/aptext/components/three_fold.py#L57-L88
nr = apt.components.AmBe()

# i should write my own stuff (or use minghao's) under components of aptext of apt
# this component tells me exactly which steps to do, just like registering cuts in cutax
# https://github.com/XENONnT/applefiles/blob/cbb3139150526e647d1e2985f2079577f8218cd3/aptext/components/two_fold.py#L121-L146

# %%
aa = nr.show_config()
# all the ingredients it needs
# if current = None, means it uses default

# %%
aa

# %%
f_param_config = '/home/puehlengt/appletree/notebooks/param_nr_sr1_3params.json'
param_flavour = '3params'
#f_param_config = '/home/puehlengt/appletree/notebooks/param_nr_sr1_short.json'
#f_param_config = '/home/puehlengt/appletree/notebooks/param_nr_sr1.json'

f_instruct = 'instruct_ambe_realistic.json'
with open(f_instruct, 'rb') as fid:
    instruct = json.load(fid) # dictionary

# %%
my_config = {} # get this from miao when using his component in the future

for _opt in aa['option']: # for everything that you can set
    if _opt in instruct['configs']: # if it's inside the instruction file
        my_config[_opt] = instruct['configs'][_opt] # use that

# %%
my_config

# %%
# Deduce the workflow(datastructure)
nr.deduce(data_names=["cs1", "cs2"], func_name="simulate")  # 'eff'(efficiency) is always simulated
# above line basically says that i want cs1, cs2, go deduce and gather ingredients required to compute those

nr.set_config(my_config) # this must be done before nr.compile
nr.rate_name = "nr_rate"  # also we have to specify a normalization factor of the component

# Compile NR script
# This is meta-programing because  appletree can generate codes dynamically
nr.compile()

# %%
#apt.clear_cache() # to clear cache, if you want to re-compile the component

# %%
# apt.share._cached_configs, apt.share._cached_functions

# %%
nr.show_config()

# %%
# For reference, this is the compiled code, the function is stored in appletree.share._cached_functions
# Initialize component
print('NR')
print(nr.code)

# %%
# nr.needed_parameters

# %% [markdown]
# ## Simulation

# %%
# Of course we have to load parameters (and their priors) in simulation (who the hell writes such comments..)
#par_manager = apt.Parameter(get_file_path(instruct['par_config'])) 
par_manager = apt.Parameter(f_param_config)

# %%
parameters = par_manager.get_all_parameter()

# %%
floated_param = []
for _par in parameters.keys():
    if isinstance(parameters[_par], np.float64):
        floated_param.append(_par)
print(f'Floating params: {floated_param}')

# %%
t_start = time()

# %%
batch_size = int(1e4) # for funsies. design flaw, batch_size doesn't do shits here for multiple scatters like AmBe

param_bag = []
events_bag = []

for _mc in range(num_sims):
    key = apt.randgen.get_key(seed=_mc+MC_ID) # for reproducibility, we can set the seed to be the same as the MC_ID, or something else. up to you.

    par_manager.sample_prior() # sampling from prior
    parameters = par_manager.get_all_parameter()

    # simulate
    key, (cs1, cs2, eff) = nr.simulate(key, batch_size, parameters)

    # randomly sampling number of events according to ambe_nr_rate parameter
    n = sps.poisson.rvs(mu=parameters['ambe_nr_rate'])

    # must normalise
    norm_eff = np.array(eff)
    norm_eff /= norm_eff.sum()

    # randomly sample n events from all sim events according to weights, norm_eff
    sel_ind = np.random.choice(len(eff), size=n, p=norm_eff, replace=False) # np reweighs weights for me
    sel_cs1, sel_cs2 = np.array(cs1[sel_ind]), np.array(cs2[sel_ind])
    events = np.vstack((sel_cs1, sel_cs2))

    # only saving parameters that were floated
    save_params = {}
    for key, val in parameters.items():
        if key in floated_param:
            save_params[key] = val.item() # convert np.float64 to python float for better json serialization

    # store things
    param_bag.append(save_params)
    events_bag.append(events)
    

# %%
t_end = time()

# %%
print(f'sims took {(t_end-t_start)/60.:.1f} minutes')

# %%
save_bag = {'param_bag': param_bag,
            'events_bag': events_bag}

# %%
save_on = True
#save_on = False

if save_on:
    np.save(f'testsims_{param_flavour}_{num_sims}sims_mcid{MC_ID}.npy', save_bag)

# %%
raise

# %% [markdown]
#

# %%
# todo:
# [done] 0. realistic parameters file
# 1. save the simulated data
# [done] 2. visualize the simulated data
# [done, i think?] 3. think of a way to sample number of events according to the rate parameter

# %%
t_wcompile = 35.3
t_wocompile = 2.6
t_compile = t_wcompile-t_wocompile

t_per_sim = (t_wcompile-t_compile)/num_sims
target = 100_000

(t_compile+t_per_sim*target)/60./60. # hours

# %%
max_plots = 10

cnt = 0
for _mc in range(num_sims):
    if cnt >= max_plots:
        break
    this_params = param_bag[_mc]
    this_events = events_bag[_mc]
    print(this_params)

    plt.figure()
    plt.hist2d(this_events[0,:], this_events[1,:],
            bins=[np.linspace(0.1, 150, 100), np.geomspace(10**2, 10**4, 100)], norm=LogNorm())
    plt.xlabel('cs1 [PE]')
    plt.ylabel('cs2 [PE]')
    plt.yscale('log')
    plt.title(_mc)

    cnt += 1

# %%
batch_size = int(1e4) # design flaw, batch_size doesn't do shits here for multiple scatters like AmBe

key = apt.randgen.get_key()
key, (cs1, cs2, eff) = nr.simulate(key, batch_size, parameters)


# length of cs1, cs2, eff the same each time (but is not the same as batch_size), but the sum of eff changes (thankfully)

# %%
num_sigmas = 6
tail_prob = sps.norm.sf(x=num_sigmas)
suggested_max_batch = sps.norm.ppf(1-tail_prob,
                                   loc=par_manager.par_config['ambe_nr_rate']['init_mean'],
                                   scale=par_manager.par_config['ambe_nr_rate']['init_mean'])
print(suggested_max_batch) # number of NR events hardly gonna fluctuate above this

# %%
plt.hist2d(sel_cs1, sel_cs2, weights=eff[sel_ind], bins=[np.linspace(0.1, 150, 100), np.geomspace(10**2, 10**4, 100)], norm=LogNorm())
plt.yscale('log')

# %%
n_events_selected = _cached_functions['AmBe_llh']['BootstrapMS_AmBe'].g4.n_events_selected # tbh what is this again?

# %%
key, (cs1, cs2, eff) = nr.simulate(key, batch_size, parameters)

# %%
target = 200_000

40/num_sims*target/60./60. # hours
