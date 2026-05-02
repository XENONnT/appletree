#!/bin/bash

num_sims=200

f_param_config='/home/puehlengt/appletree/notebooks/param_nr_sr1_6params.json'
param_flavour='6params'
#f_param_config='/home/puehlengt/appletree/notebooks/param_nr_sr1_3params.json'
#param_flavour='3params'

# Loop from 1 to 10
#for ii in {1..10}
for ii in {10001..10010}
do
    echo "Running with argument: $ii"
    python sim_nr_realistic.py $ii $num_sims $f_param_config $param_flavour > "apt_${param_flavour}_${num_sims}sims_mcind${ii}.log" 2>&1 &
done

wait
echo "All jobs done!"
