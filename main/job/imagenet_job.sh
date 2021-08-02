#!/bin/bash -l

#SBATCH -p datasci


#SBATCH --job-name=imagenet_train
##SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
cd ..

# scd01


python train_mlp_major_.py --dataset imagenet --n_classes 2 --iters 200 --hidden-nodes 20 --lr 0.01 --save --seed 2018 \
--round 100 --target imagenet_100_mlp_logistic_20.pkl

#python train_lenet_.py --dataset imagenet --seed 2018 --n_classes 2 \
#--save --target imagenet_resnet50_1.pkl --round 1
#
#python train_lenet_.py --dataset imagenet --seed 2018 --n_classes 2 \
#--save --target imagenet_lenet_100.pkl --round 100

#python train_binarynn.py --dataset imagenet --n_classes 2 --hidden-nodes 20 --seed 2018 \
#--target imagenet_mlpbnn_approx --round 100

