#!/bin/bash

source /home/taha/miniconda3/etc/profile.d/conda.sh

conda remove --name mmdetection -y --all
rm -rf /home/taha/miniconda3/envs/mmdetection

conda create -n mmdetection -y
conda activate mmdetection

conda install python=3.7 pytorch torchvision cudatoolkit=11 -c pytorch-nightly -y

# install the latest mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

# install mmdetection
cd mmdetection
pip install -r requirements/build.txt
pip install -e .
cd ..

python -c 'from mmdet.apis import init_detector, inference_detector'

conda deactivate