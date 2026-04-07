#!/bin/bash


name="alexnet"
datashape="(224,224,3)"


archit_in="subclassing"
archit_out="subclassing"


python tf2torch/tf2torch_migrator.py "output/${name}/tf_nn_${archit_in}.py" ${archit_in} ${archit_out} --datashape ${datashape}

