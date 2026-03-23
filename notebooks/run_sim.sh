#!/bin/bash

# Loop from 1 to 10
for ii in {1..10}
do
    echo "Running with argument: $ii"
    python sim_nr_realistic.py $ii > "apt_sims_mcind${ii}.log" 2>&1 &
done

wait
echo "All jobs done!"
