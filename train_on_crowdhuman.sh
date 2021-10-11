#!/bin/bash

#SBATCH -J yolof
#SBATCH --gres=gpu:8
#SBATCH --mem=250GB
#SBATCH -N 1
#SBATCH -A laszka
#SBATCH -t 96:00:00

source /project/cacds/apps/anaconda3/5.0.1/etc/profile.d/conda.sh

conda activate mmdetection

cd /home/teghtesa/carta-image-processing

python mmdetection/tools/dist_train.sh new_yolof_config.py 8