#!/bin/bash

#SBATCH -J yolof
#SBATCH --gres=gpu:8
#SBATCH --mem=250GB
#SBATCH -N 1
#SBATCH -A laszka
#SBATCH -t 96:00:00

echo "Running training for $1"

source /project/cacds/apps/anaconda3/5.0.1/etc/profile.d/conda.sh

conda activate mmdetection

cd /home/teghtesa/carta-image-processing || exit 255

bash mmdetection/tools/dist_train.sh "$1" 8