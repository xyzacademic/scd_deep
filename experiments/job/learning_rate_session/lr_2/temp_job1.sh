#!/bin/bash -l
#SBATCH -p datasci3,datasci4,datasci
#SBATCH -x node429,node430,node415
#SBATCH --job-name=cifar10_01
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
cd ..
gpu=0

echo cifar10_toy3ssr100_sign_i1_mce_b1500_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_0
python train_cnn01_ce.py --nrows 0.03 --localit 1 --updated_conv_features 32 --updated_fc_features 128 --updated_fc_nodes 1 --updated_conv_nodes 1 --width 100 --normalize 0 --percentile 1 --fail_count 1 --loss mce --act sign --fc_diversity 1 --init normal --no_bias 2 --scale 1 --w-inc1 0.05 --w-inc2 0.1 --version toy3ssr100 --seed 0 --iters 15000 --dataset cifar10 --n_classes 10 --cnn 1 --divmean 0 --target cifar10_toy3ssr100_sign_i1_mce_b1500_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_0 --updated_fc_ratio 5 --updated_conv_ratio 10 --verbose_iter 20 --freeze_layer 1 --fp16 --temp_save_per_iter 1000 --resume --source cifar10_toy3srr100_sign_i1_mce_b1500_lrc0.075_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_0 
