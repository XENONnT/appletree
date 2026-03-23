#!/bin/bash

num_sims=5

# Loop from 1 to 10
for ii in {1..10}
do
    echo "Running with argument: $ii"
    python sim_nr_realistic.py $ii $num_sims> "apt_10params_${num_sims}sims_mcind${ii}.log" 2>&1 &
done

wait
echo "All jobs done!"
