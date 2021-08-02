#!/bin/bash -l
#SBATCH -p datasci3,datasci4
#SBATCH --job-name=cifar10_01
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
cd ..
gpu=0
cpus=2

python train_bce.py --aug 1 --n_classes 2 --no_bias 1 --seed 2 --version toy3ssr100scale --lr 0.001 --target cifar10_binary_89_toy3ssr100scale_nb2_bce_bp_2 --batch-size 256 --dataset cifar10_binary --c0 8 --c1 9 --epoch 100 --save
