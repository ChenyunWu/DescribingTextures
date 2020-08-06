#!/usr/bin/env bash

cd ~/work1/DescribeTexture
. /home/chenyun/anaconda3/etc/profile.d/conda.sh
conda activate py37_cuda92

exp=bc
exp_i=0
partition=titanx-long
for input in phrase description; do
  for encoder in mean lstm elmo bert; do
#    if ((exp_i==0 || exp_i==1 || exp_i==4 || exp_i==5)); then
      out="output/triplet_match/${exp}${exp_i}_${input}_${encoder}"
      options="LANG_INPUT ${input}
      MODEL.LANG_ENCODER ${encoder}
      OUTPUT_PATH ${out}
      "
      echo "${exp}${exp_i}: ${out}"
      export options
      fname=${exp}${exp_i}
      sbatch --job-name tm_${fname} -o logs/tm_train_${fname}.out -e logs/tm_train_${fname}.err \
              -p ${partition} --exclude node035 models/triplet_match/train.sbatch
      sleep 0.5
#    fi
    ((exp_i=exp_i+1))
  done
done