#!/bin/bash
. /home/kdas/anaconda3/etc/profile.d/conda.sh
conda info --envs
conda activate crysgnn_gpu
python -V
conda info --envs
nvidia-smi
cd /archive/pascal/kdas/crysgnn_test/distilled_baslines/cgcnn_distilled/
pwd
python train.py --epochs 1000 --data-path '../processed_data/jarvis/optb88vdw_total_energy/' --lr 0.003 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
#python train.py --epochs 1000 --data-path '../processed_data/mp_2018/bgap/' --lr 0.003 --optim 'Adam' --train-size 60000 --val-size 5000 --test-size 4239


