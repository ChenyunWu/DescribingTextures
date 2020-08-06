#!/usr/bin/env bash

cd ~/work1/DescribeTexture
. /home/chenyun/anaconda3/etc/profile.d/conda.sh
conda activate py37_cuda92

exp=a
exp_i=0
partition=1080ti-long
for wd in 1e-4 1e-6; do
  for lr in 0.01 0.001 0.0001; do
    for margin in 1.0 10.0 20.0; do
      for ph_w in 0.2 1.0 5.0; do
        if ((exp_i >=40));then
          partition=titanx-long
        fi
          out="output/triplet_match/${exp}${exp_i}_lr${lr}_wd${wd}_margin${margin}_phw${ph_w}"
          options="TRAIN.INIT_LR ${lr}
          TRAIN.WEIGHT_DECAY ${wd}
          OUTPUT_PATH ${out}
          LOSS.MARGIN ${margin}
          LOSS.IMG_SENT_WEIGHTS (1.0,${ph_w})
          "
          echo "${exp}${exp_i}: ${out}"
          export options
          fname=${exp}${exp_i}
          sbatch --job-name tm_${fname} -o logs/tm_train_${fname}.out -e logs/tm_train_${fname}.err \
                  -p ${partition} --exclude node080 models/triplet_match/train.sbatch
          sleep 0.5
        ((exp_i=exp_i+1))
      done
    done
  done
done