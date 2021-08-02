#!/bin/bash -l

#SBATCH -p datasci3
#SBATCH --job-name=mnist_train
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#cvb


cd ..
# scd01

# 10 015 0.5
#

nohup python train_scd_vote.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 10 --b-ratio 0.2 \
--updated-features 64 --round 4 --interval 10 --gpu 0  --n-jobs 4 --num-gpus 4 --save --n_classes 2 --iters 1 \
--target mnist_k1.pkl --dataset mnist  --seed 1 > temp_logs &


python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 10 --b-ratio 0.2 \
--updated-features 64 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
--target mnist_k1.pkl --dataset mnist  --seed 1 &

python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 20 --b-ratio 0.2 \
--updated-features 64 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
--target mnist_k1.pkl --dataset mnist  --seed 2018 --c 0.1 >> temp_logs1 &

python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 10 --b-ratio 0.2 \
--updated-features 64 --round 1 --interval 10 --gpu 1  --save --n_classes 2 --iters 1 \
--target mnist_k2.pkl --dataset mnist  --seed 2018 --c 0.2 >> temp_logs2 &

python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 10 --b-ratio 0.2 \
--updated-features 64 --round 1 --interval 10 --gpu 2  --save --n_classes 2 --iters 1 \
--target mnist_k3.pkl --dataset mnist  --seed 2018 --c 0.3 >> temp_logs3 &

python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 10 --b-ratio 0.2 \
--updated-features 64 --round 1 --interval 10 --gpu 3  --save --n_classes 2 --iters 1 \
--target mnist_k4.pkl --dataset mnist  --seed 2018 --c 0.4 >> temp_logs4 &

#python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 10 --b-ratio 0.2 \
#--updated-features 64 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target mnist_scd01_1_br02_nr075_ni1000_i1.pkl --dataset mnist  --seed 2
#
#python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 10 --b-ratio 0.2 \
#--updated-features 64 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target mnist_scd01_1_br02_nr075_ni1000_i1.pkl --dataset mnist  --seed 3
#
#python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 10 --b-ratio 0.2 \
#--updated-features 64 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target mnist_scd01_1_br02_nr075_ni1000_i1.pkl --dataset mnist  --seed 4
#
#python train_mlp01.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --num-iters 10 --hidden-nodes 20 --b-ratio 0.2 \
#--updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target mnist_mlp01_1_br02_nr075_ni1000_i1.pkl --dataset mnist  --seed 2018