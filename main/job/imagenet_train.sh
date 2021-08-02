#!/bin/bash -l

#SBATCH -p datasci


#SBATCH --job-name=imagenet_job
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

cd ..







#python train_scd4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 \
#--updated-features 256 --round 100 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 2 --iters 1 \
#--target imagenet_scd01mlp_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset imagenet --version mlp --seed 0 --width \
#10000 \
#--metrics balanced --init normal  --updated-nodes 1

#python train_scd4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 256 --round 100 --interval 20 --n-jobs 2 --num-gpus 2  --save --n_classes 2 --iters 1 \
#--target imagenet_scdcemlp_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset imagenet --version ce --seed 0 --width 10000 \
#--metrics balanced --init normal  --updated-nodes 1

python train_scd4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 10000 \
--b-ratio 0.2 --updated-features 256 --round 100 --interval 20 --n-jobs 2 --num-gpus 2  --save --n_classes 2 \
--iters 1 --target imagenet_scd01mlpbnn_100_br02_h20_nr075_ni10000_i1_0.pkl --dataset imagenet --version 01bnn --seed 0 \
--width 10000 --metrics balanced --init normal  --updated-nodes 1

python train_scd4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 10000 \
--b-ratio 0.2 --updated-features 256 --round 100 --interval 20 --n-jobs 2 --num-gpus 2  --save --n_classes 2 \
--iters 1 --target imagenet_scdcemlpbnn_100_br02_h20_nr075_ni10000_i1_0.pkl --dataset imagenet --version cebnn --seed 0 \
--width 10000 --metrics balanced --init normal  --updated-nodes 1