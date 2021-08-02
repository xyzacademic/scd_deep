#!/bin/bash -l
#SBATCH -p datasci,datasci3,datasci4
#SBATCH --job-name=cifar10_01
#SBATCH -x node429,node430
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
cd ..
gpu=0

python train_cnn01.py --nrows 0.75 --localit 1 --updated_conv_features 32 --updated_fc_features 128 --updated_fc_nodes 1 --updated_conv_nodes 1 --width 100 --normalize 1 --percentile 1 --fail_count 1 --loss bce --act sigmoid --fc_diversity 1 --init normal --no_bias 1 --scale 1 --w-inc1 0.17 --w-inc2 0.2 --version fc --seed 12 --iters 1000 --dataset cifar10 --n_classes 2 --cnn 0 --divmean 0 --target cifar10_fc_sigmoid_i1_bce_nb1_nw1_dm0_s1_fp32_32_12.pkl --updated_fc_ratio 1 --updated_conv_ratio 1
