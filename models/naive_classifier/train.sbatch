#!/usr/bin/env bash
# train.sbatch
#
#SBATCH --gres=gpu:1
##SBATCH -p 1080ti-short               # Partition to submit to (serial_requeue)
#SBATCH -N 1                          # Ensure that all cores are on one machine
##SBATCH --mem=10240                     # Memory pool for all cores (see also --mem-per-cpu)
##SBATCH --mail-type=FAIL               # Type of email notification- BEGIN,END,FAIL,ALL
##SBATCH --mail-user=chenyun@cs.umass.edu # Email to which notifications will be sent

srun python models/naive_classifier/train.py -o ${options}