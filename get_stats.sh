#!/bin/bash

exp_id=$1
save_dir=$2
ckpt_num=$3

echo "Measuring network latency using a fixed number of inputs.."
python blueoil/cmd/measure_latency.py -i $exp_id

echo "Profiling the network for getting network size.."
python blueoil/cmd/profile_model.py -i $exp_id

echo "Exporting the model for getting the protocol buffer for network graph.."
python blueoil/cmd/export.py -i $1 --restore_path $2/$1/checkpoints/save.ckpt-$3
