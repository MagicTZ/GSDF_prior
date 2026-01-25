#!/bin/bash
exp_dir=./exp
config=configs/scannetpp/0a184cf634.yaml
gpu=0
tag=with_prior

python launch.py \
    --exp_dir ${exp_dir} \
    --config ${config} \
    --gpu ${gpu} \
    --train \
    --eval \
    tag=${tag}
