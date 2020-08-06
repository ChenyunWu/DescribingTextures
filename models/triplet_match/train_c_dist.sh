#!/usr/bin/env bash

cd ~/work1/DescribeTexture
. /home/chenyun/anaconda3/etc/profile.d/conda.sh
conda activate py37_cuda92

exp=c
exp_i=0
partition=1080ti-long

for encoder in mean lstm elmo bert; do
  for lr in 0.0001 0.00003 0.00001; do
    for dist in l2 l2_s cos; do
      margin=1.0
      if [ ${dist} == cos ]; then
        margin=0.5
#      fi

#      if [ ${dist} == l2 ]; then
        out="output/triplet_match/${exp}${exp_i}_${encoder}_${dist}_lr${lr}"
        options="OUTPUT_PATH ${out}
        MODEL.LANG_ENCODER ${encoder}
        MODEL.DISTANCE ${dist}
        LOSS.MARGIN ${margin}
        TRAIN.INIT_LR ${lr}
        "
        echo "${exp}${exp_i}: ${out} ${margin}"
        export options
        fname=${exp}${exp_i}
        sbatch --job-name tm_${fname} -o logs/tm_train_${fname}.out -e logs/tm_train_${fname}.err \
                -p ${partition} --exclude node035 models/triplet_match/train.sbatch
        sleep 0.5
      fi

      ((exp_i=exp_i+1))
    done
  done
done