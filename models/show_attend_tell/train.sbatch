#!/usr/bin/env bash
# train.sbatch
#
#SBATCH --job-name cap_v2
#SBATCH --gres=gpu:1
#SBATCH -p 2080ti-long               # Partition to submit to (serial_requeue)
#SBATCH -o logs/caption_train_v2.out
#SBATCH -e logs/caption_train_v2.err
#SBATCH -N 1                          # Ensure that all cores are on one machine
##SBATCH --mem=10240                     # Memory pool for all cores (see also --mem-per-cpu)
##SBATCH --mail-type=FAIL               # Type of email notification- BEGIN,END,FAIL,ALL
##SBATCH --mail-user=chenyun@cs.umass.edu # Email to which notifications will be sent

srun python models/show_attend_tell/train.py