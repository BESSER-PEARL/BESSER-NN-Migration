#!/bin/bash


name="tf_tutorial"
datashape="(32,32,3)"


archit_in="subclassing"
archit_out="subclassing"


python tf2torch/tf2torch_migrator.py "output/${name}/tf_nn_${archit_in}.py" ${archit_in} ${archit_out} --datashape ${datashape}


output_file="max_abs_diffs_${name}_tf2torch_${archit_in}_${archit_out}.txt"
> "$output_file"  # Clear previous content

for i in {1..100}
do
    echo "Run #$i"
    # Run pytest, capture stdout
    # Extract line with "Max absolute difference:" and grab the number
    max_diff=$(pytest -p no:warnings -s -q test_functional_behavior.py --fpytorch output/migrated_nn/pytorch_nn_${archit_out}.py  --ftensorflow output/${name}/tf_nn_${archit_in}.py --datashape ${datashape} | grep "Max absolute difference:" | awk '{print $4}')
    
    if [[ -n "$max_diff" ]]; then
        echo "$max_diff" >> "$output_file"
    else
        echo "No max diff found in run #$i" >> "$output_file"
    fi
done

echo "All max absolute differences saved in $output_file"
