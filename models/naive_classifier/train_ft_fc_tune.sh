#!/usr/bin/env bash

cd ~/work1/DescribeTexture
. /home/chenyun/anaconda3/etc/profile.d/conda.sh
conda activate py37_cuda92

exp=v1
exp_i=0
partition=1080ti-short
for tune in False True; do
    for fc in "" "512"; do
        for ft in "4" "3" "2" "1" "1,4" "2,4" "3,4" "1,3" "2,3" "1,2"; do
            if ((exp_i >= 7));then
              if ((exp_i <= 19));then
                partition=2080ti-short

            out="output/naive_classify/${exp}_${exp_i}_ft${ft}_fc${fc}_tune${tune}"
            options="MODEL.BACKBONE_FEATS [${ft}] MODEL.FC_DIMS [${fc}] TRAIN.TUNE_BACKBONE ${tune} OUTPUT_PATH ${out}"
            echo "${exp}${exp_i}: ${options}"
            export options
            fname=nc_${exp}_${exp_i}
            sbatch --job-name ${fname} --exclude node155 -p ${partition} -o logs/train_${fname}.out -e logs/train_${fname}.err \
                    models/naive_classifier/train.sbatch
            sleep 0.1
            fi
            fi
           ((exp_i=exp_i+1))
        done
    done
done