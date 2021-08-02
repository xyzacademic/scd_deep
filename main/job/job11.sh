#!/bin/sh

#SBATCH -p datasci


#SBATCH --job-name=text_job
##SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2

cd ..

data='imdb'
gpus=2
# mlp logistic
#python train_mlp_major_.py --dataset ${data} --n_classes 2 --hidden-nodes 20 --lr 0.1 --iters 1000 --seed 2018 \
#--save --target ${data}_mlp_1.pkl --round 1
#
#python train_mlp_major_.py --dataset ${data} --n_classes 2 --hidden-nodes 20 --lr 0.1 --iters 1000 --seed 2018 \
#--save --target ${data}_mlp_8.pkl --round 8
#
#python train_mlp_major_.py --dataset ${data} --n_classes 2 --hidden-nodes 20 --lr 0.1 --iters 1000 --seed 2018 \
#--save --target ${data}_mlp_100.pkl --round 100

# scd ce

#python train_scd4.py --nrows 0.15 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 8 --interval 20 --n-jobs ${gpus} --num-gpus ${gpus}  --save --n_classes 2 --iters 1 \
#--target ${data}_scdcemlp_8_br02_h20_nr075_ni1000_i1_0.pkl --dataset ${data} --version ce --seed 0 --width 10000 \
#--metrics balanced --init normal  --updated-nodes 1
#
#python train_scd4.py --nrows 0.15 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 100 --interval 20 --n-jobs ${gpus} --num-gpus ${gpus}  --save --n_classes 2 --iters 1 \
#--target ${data}_scdcemlp_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset ${data} --version ce --seed 0 --width 10000 \
#--metrics balanced --init normal  --updated-nodes 1

# random forest
#python train_dt.py --dataset ${data} --n_classes 2 --round 100 --save --target ${data}_rf_100.pkl --seed 2018

python train_dt.py --dataset ${data} --n_classes 2 --round 8 --save --target ${data}_rf_8.pkl --seed 2018

## mlp01
#python train_scd4.py --nrows 0.15 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 8 --interval 20 --n-jobs ${gpus} --num-gpus ${gpus}  --save --n_classes 2 --iters 1 \
#--target ${data}_scd01mlp_8_br02_h20_nr075_ni1000_i1_0.pkl --dataset ${data} --version mlp --seed 0 --width 10000 \
#--metrics balanced --init normal  --updated-nodes 1
#
#python train_scd4.py --nrows 0.15 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 100 --interval 20 --n-jobs ${gpus} --num-gpus ${gpus}  --save --n_classes 2 --iters 1 \
#--target ${data}_scd01mlp_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset ${data} --version mlp --seed 0 --width 10000 \
#--metrics balanced --init normal  --updated-nodes 1

## mlp01bnn
#python train_scd4.py --nrows 0.15 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 8 --interval 20 --n-jobs ${gpus} --num-gpus ${gpus}  --save --n_classes 2 --iters 1 \
#--target ${data}_scd01mlpbnn_8_br02_h20_nr075_ni1000_i1_0.pkl --dataset ${data} --version 01bnn --seed 0 --width 10000 \
#--metrics balanced --init normal  --updated-nodes 1
#
#python train_scd4.py --nrows 0.15 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 100 --interval 20 --n-jobs ${gpus} --num-gpus ${gpus}  --save --n_classes 2 --iters 1 \
#--target ${data}_scd01mlpbnn_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset ${data} --version 01bnn --seed 0 --width 10000 \
#--metrics balanced --init normal  --updated-nodes 1

# mlp ce bnn

#python train_scd4.py --nrows 0.15 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 8 --interval 20 --n-jobs ${gpus} --num-gpus ${gpus}  --save --n_classes 2 --iters 1 \
#--target ${data}_scdcemlpbnn_8_br02_h20_nr075_ni1000_i1_0.pkl --dataset ${data} --version cebnn --seed 0 --width 10000 \
#--metrics balanced --init normal  --updated-nodes 1
#
#python train_scd4.py --nrows 0.15 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 100 --interval 20 --n-jobs ${gpus} --num-gpus ${gpus}  --save --n_classes 2 --iters 1 \
#--target ${data}_scdcemlpbnn_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset ${data} --version cebnn --seed 0 --width 10000 \
#--metrics balanced --init normal  --updated-nodes 1


## bnn
#
#python train_binarynn.py --dataset ${data} --n_classes 2 --hidden-nodes 20 --seed 2018 \
#--target ${data}_mlpbnn_approx --round 100