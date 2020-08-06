#!/usr/bin/env bash

cd ~/work1/DescribeTexture
. /home/chenyun/anaconda3/etc/profile.d/conda.sh
conda activate py37_cuda92

exp=da
exp_i=0
partition=2080ti-long

for encoder in mean lstm elmo bert; do
  lr=0.00003
  if [ ${encoder} == bert ]; then
    lr=0.00001
  fi
#  if [ $exp_i == 0 ]; then
    out="output/triplet_match/${exp}${exp_i}_${encoder}_lr${lr}"
    options="OUTPUT_PATH ${out}
    LANG_INPUT description
    TRAIN.MAX_EPOCH 30
    TRAIN.EVAL_EVERY_EPOCH 0.2
    TRAIN.CHECKPOINT_EVERY_EPOCH 1.0
    MODEL.LANG_ENCODER ${encoder}
    TRAIN.INIT_LR ${lr}
    "
    echo "${exp}${exp_i}: ${out}"
    export options
    fname=${exp}${exp_i}
    sbatch --job-name tm_${fname} -o logs/tm_train_${fname}.out -e logs/tm_train_${fname}.err \
            -p ${partition} --exclude node163 models/triplet_match/train.sbatch
    sleep 0.5
#  fi
  ((exp_i=exp_i+1))

done