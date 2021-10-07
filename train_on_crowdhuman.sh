#!/bin/bash

#SBATCH --gres=gpu:8
#SBATCH --mem=250GB
#SBATCH -N 1
#SBATCH -A laszka

source /project/cacds/apps/anaconda3/5.0.1/etc/profile.d/conda.sh

conda activate mmdetection

cd /home/teghtesa/carta-image-processing

python mmdetection/tools/train.py new_yolof_config.py