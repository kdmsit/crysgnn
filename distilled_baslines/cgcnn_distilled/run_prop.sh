#!/bin/bash
. /home/kdas/anaconda3/etc/profile.d/conda.sh
conda info --envs
conda activate crysgnn_gpu
python -V
conda info --envs
nvidia-smi
cd /archive/pascal/kdas/crysgnn_test/distilled_baslines/cgcnn_distilled/

# JARVIS
python train.py --epochs 1000 --data-path '../processed_data/jarvis/formation_energy_peratom/' --lr 0.003 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
python train.py --epochs 1000 --data-path '../processed_data/jarvis/optb88vdw_bandgap/' --lr 0.003 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
python train.py --epochs 1000 --data-path '../processed_data/jarvis/optb88vdw_total_energy/' --lr 0.003 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
python train.py --epochs 1000 --data-path '../processed_data/jarvis/bulk_modulus_kv/' --lr 0.003 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
python train.py --epochs 1000 --data-path '../processed_data/jarvis/shear_modulus_gv/' --lr 0.003 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
python train.py --epochs 1000 --data-path '../processed_data/jarvis/mbj_bandgap/' --lr 0.003 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
python train.py --epochs 1000 --data-path '../processed_data/jarvis/slme/' --lr 0.003 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
python train.py --epochs 1000 --data-path '../processed_data/jarvis/spillage/' --lr 0.003 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
python train.py --epochs 1000 --data-path '../processed_data/jarvis/ehull/' --lr 0.003 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
python train.py --epochs 1000 --data-path '../processed_data/jarvis/epsx/' --lr 0.003 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
python train.py --epochs 1000 --data-path '../processed_data/jarvis/epsy/' --lr 0.003 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
python train.py --epochs 1000 --data-path '../processed_data/jarvis/epsz/' --lr 0.003 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
python train.py --epochs 1000 --data-path '../processed_data/jarvis/mepsx/' --lr 0.003 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
python train.py --epochs 1000 --data-path '../processed_data/jarvis/mepsy/' --lr 0.003 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
python train.py --epochs 1000 --data-path '../processed_data/jarvis/mepsz/' --lr 0.003 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
python train.py --epochs 1000 --data-path '../processed_data/jarvis/n-powerfact/' --lr 0.003 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
python train.py --epochs 1000 --data-path '../processed_data/jarvis/p-powerfact/' --lr 0.003 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
python train.py --epochs 1000 --data-path '../processed_data/jarvis/n-Seebeck/' --lr 0.003 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
python train.py --epochs 1000 --data-path '../processed_data/jarvis/p-Seebeck/' --lr 0.003 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

#MP-2018
python train.py --epochs 1000 --data-path '../processed_data/mp_2018/bgap/' --lr 0.003 --optim 'Adam' --train-size 60000 --val-size 5000 --test-size 4239
python train.py --epochs 1000 --data-path '../processed_data/mp_2018/delta/' --lr 0.003 --optim 'Adam' --train-size 60000 --val-size 5000 --test-size 4239


