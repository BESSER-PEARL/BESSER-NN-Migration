#!/bin/bash


name="alexnet"
datashape="(224,224,3)"


archit_in="subclassing"
archit_out="subclassing"


python torch2tf/torch2tf_migrator.py "output/${name}/pytorch_nn_${archit_in}.py" ${archit_in} ${archit_out}


output_file="max_abs_diffs_${name}_torch2tf_${archit_in}_${archit_out}.txt"
> "$output_file"  # Clear previous content

for i in {1..100}
do
    echo "Run #$i"
    # Run pytest, capture stdout
    # Extract line with "Max absolute difference:" and grab the number
    max_diff=$(pytest -p no:warnings -s -q test_functional_behavior.py --ftensorflow output/migrated_nn/tf_nn_${archit_out}.py  --fpytorch output/${name}/pytorch_nn_${archit_in}.py --datashape ${datashape} | grep "Max absolute difference:" | awk '{print $4}')
    
    if [[ -n "$max_diff" ]]; then
        echo "$max_diff" >> "$output_file"
    else
        echo "No max diff found in run #$i" >> "$output_file"
    fi
done

echo "All max absolute differences saved in $output_file"

