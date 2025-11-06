import numpy as np
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import multihist as mh

import GOFevaluation

import appletree as apt
from appletree.utils import load_json, get_file_path

apt.set_gpu_memory_usage(0.3)

# Load configuration file
config = get_file_path("xe131m.json")
config = load_json(config)
filename = '/home/ykaminaga/gamma_apt/fit_result/xe131m_sr1_fit_result_0703_50_20000.h5'
config['backend_h5'] = filename
# Initialize context
tree = apt.Context(config)

tree.print_context_summary(short=True)

result = tree.fitting(nwalkers=50, iteration=20000)

logp = tree.sampler.get_log_prob()

for _logp in logp.T:
    plt.plot(_logp, lw=0.1)

plt.xlabel("iteration")
plt.ylabel("log posterior")
plt.savefig("notebooks/fig/Log_posterior_xe131m_sr1_fit_result_0703_50_20000.SVG")
plt.savefig("notebooks/fig/Log_posterior_xe131m_sr1_fit_result_0703_50_20000.jpg")

cs1, cs2, eff = tree.get_template("xe131m_llh", "xe131m_er")

h, be = jnp.histogramdd(
    jnp.asarray([cs1, cs2]).T,
    bins=(jnp.linspace(1000, 1700, 101), jnp.linspace(10000, 60000, 101)),
    weights=eff,
)

h = mh.Histdd.from_histogram(np.array(h), be, axis_names=["cs1", "cs2"])
h.plot(norm=LogNorm())
plt.scatter(*tree["xe131m_llh"].data.T, color="r", s=2.0)
plt.savefig("notebooks/fig/cs12_xe131m_sr1_fit_result_0703_50_20000.jpg")

parameters = tree.get_post_parameters()
key = apt.randgen.get_key()
batch_size = int(1e6)
print(parameters)
with open('/home/ykaminaga/appletree/appletree/parameters/er_xe131m.json', 'w') as f:
    json.dump(parameters, f, indent=2)

import h5py
import json
from appletree import Plotter
fig = Plotter(filename, thin=2)
fig.make_all_plots()
plt.savefig("notebooks/fig/All_plots_xe131m_sr1_constantr_fit_result_0702_50_20000.jpg")
plt.savefig("notebooks/fig/All_plots_xe131m_sr1_constantr_fit_result_0702_50_20000.SVG");
