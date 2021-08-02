#!/bin/bash -l
#SBATCH -p datasci3,datasci4
#SBATCH --job-name=mlp01scd
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
cd ..
gpu=0
cpus=2

python train_cnn01_01.py --nrows 0.75 --localit 1 --updated_fc_features 128 --updated_fc_nodes 1 --width 100 --normalize 1 --percentile 1 --fail_count 1 --loss 01loss --act sign --fc_diversity 1 --init normal --no_bias 0 --scale 1 --w-inc1 0.17 --w-inc2 0.3 --version mlp01scale --seed 0 --iters 1000 --dataset cifar10_binary --n_classes 2 --cnn 0 --divmean 0 --target cifar10_binary_01_lr0.3_mlp01scd_0 --updated_fc_ratio 1 --verbose_iter 1 --c0 0 --c1 1 
 