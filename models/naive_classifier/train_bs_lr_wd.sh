#!/usr/bin/env bash

cd ~/work1/DescribeTexture
. /home/chenyun/anaconda3/etc/profile.d/conda.sh
conda activate py37_cuda92

exp=bs
exp_i=0
for wd in 1e-4 1e-6; do
    for bs in 16 32 64; do
        for lr in 0.0003 0.0001 0.00003; do
              out="output/naive_classify/${exp}${exp_i}_lr${lr}_bs${bs}_wd${wd}"
              options="TRAIN.INIT_LR ${lr} TRAIN.WEIGHT_DECAY ${wd} TRAIN.BATCH_SIZE ${bs} OUTPUT_PATH ${out}"
              echo "${exp}${exp_i}: ${options}"
              export options
              fname=nc_${exp}${exp_i}
              sbatch --job-name ${fname} -o logs/train_${fname}.out -e logs/train_${fname}.err \
                      models/naive_classifier/train.sbatch
              sleep 0.1
              ((exp_i=exp_i+1))
        done
    done
done