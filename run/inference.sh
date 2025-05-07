#!/bin/sh
set -x

exp_dir=$1
config=$2
foldername=$3
feature_type=$4

mkdir -p ${exp_dir}
result_dir=${exp_dir}/result_inference

export PYTHONPATH=.
python -u run/inference.py \
  --config=${config} \
  foldername ${foldername} \
  feature_type ${feature_type} \
  save_folder ${result_dir} \
  2>&1 | tee -a ${exp_dir}/inference-$(date +"%Y%m%d_%H%M").log