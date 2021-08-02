#!/bin/sh

#SBATCH -p datasci

#SBATCH --job-name=gtsrb_binary_job
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
cd ..

# scd01

gpu=0

#python train_scd4.py --nrows 0.25 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 256 --round 100 --interval 20 --n-jobs 2 --num-gpus 2  --save --n_classes 2 --iters 1 \
#--target gtsrb_binary_scdcemlp_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset gtsrb_binary --version ce --seed 0 --width 10000 \
#--metrics balanced --init normal  --updated-nodes 1

python train_scd4.py --nrows 0.25 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 10000 \
--b-ratio 0.2 --updated-features 256 --round 100 --interval 20 --n-jobs 2 --num-gpus 2  --save --n_classes 2 \
--iters 1 --target gtsrb_binary_scd01mlpbnn_100_br02_h20_nr075_ni10000_i1_0.pkl --dataset gtsrb_binary --version 01bnn --seed 0 \
--width 10000 --metrics balanced --init normal  --updated-nodes 1

python train_scd4.py --nrows 0.25 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 10000 \
--b-ratio 0.2 --updated-features 256 --round 100 --interval 20 --n-jobs 2 --num-gpus 2  --save --n_classes 2 \
--iters 1 --target gtsrb_binary_scdcemlpbnn_100_br02_h20_nr075_ni10000_i1_0.pkl --dataset gtsrb_binary --version cebnn --seed 0 \
--width 10000 --metrics balanced --init normal  --updated-nodes 1

#python train_lenet_.py --dataset gtsrb_binary --n_classes 2 --round 100 --seed 2018 \
#--target gtsrb_binary_lenet_100.pkl --save
#
#python train_lenet_.py --dataset gtsrb_binary --n_classes 2 --round 100 --seed 2018 \
#--target gtsrb_binary_lenet_100_ep1.pkl --save --normal-noise --epsilon 0.1 --save
#
#python train_lenet_.py --dataset gtsrb_binary --n_classes 2 --round 100 --seed 2018 \
#--target gtsrb_binary_lenet_100_ep2.pkl --save --normal-noise --epsilon 0.2 --save
#
#python train_lenet_.py --dataset gtsrb_binary --n_classes 2 --round 100 --seed 2018 \
#--target gtsrb_binary_lenet_100_ep5.pkl --save --normal-noise --epsilon 0.5 --save
#
#python train_lenet_.py --dataset gtsrb_binary --n_classes 2 --round 100 --seed 2018 \
#--target gtsrb_binary_simplenet_100.pkl --save
#
#python train_lenet_.py --dataset gtsrb_binary --n_classes 2 --round 100 --seed 2018 \
#--target gtsrb_binary_simplenet_100_ep1.pkl --save --normal-noise --epsilon 0.1 --save
#
#python train_lenet_.py --dataset gtsrb_binary --n_classes 2 --round 100 --seed 2018 \
#--target gtsrb_binary_simplenet_100_ep2.pkl --save --normal-noise --epsilon 0.2 --save
#
#python train_lenet_.py --dataset gtsrb_binary --n_classes 2 --round 100 --seed 2018 \
#--target gtsrb_binary_simplenet_100_ep5.pkl --save --normal-noise --epsilon 0.5 --save
#
#
#python train_lenet_.py --dataset celeba --n_classes 2 --round 100 --seed 2018 \
#--target celeba_lenet_100.pkl --save
#
#python train_lenet_.py --dataset celeba --n_classes 2 --round 100 --seed 2018 \
#--target celeba_lenet_100_ep1.pkl --save --normal-noise --epsilon 0.1 --save
#
#python train_lenet_.py --dataset celeba --n_classes 2 --round 100 --seed 2018 \
#--target celeba_lenet_100_ep2.pkl --save --normal-noise --epsilon 0.2 --save
#
#python train_lenet_.py --dataset celeba --n_classes 2 --round 100 --seed 2018 \
#--target celeba_lenet_100_ep5.pkl --save --normal-noise --epsilon 0.5 --save
#
#python train_lenet_.py --dataset celeba --n_classes 2 --round 100 --seed 2018 \
#--target celeba_simplenet_100.pkl --save
#
#python train_lenet_.py --dataset celeba --n_classes 2 --round 100 --seed 2018 \
#--target celeba_simplenet_100_ep1.pkl --save --normal-noise --epsilon 0.1 --save
#
#python train_lenet_.py --dataset celeba --n_classes 2 --round 100 --seed 2018 \
#--target celeba_simplenet_100_ep2.pkl --save --normal-noise --epsilon 0.2 --save
#
#python train_lenet_.py --dataset celeba --n_classes 2 --round 100 --seed 2018 \
#--target celeba_simplenet_100_ep5.pkl --save --normal-noise --epsilon 0.5 --save


#python train_binarynn.py --dataset gtsrb_binary --n_classes 2 --hidden-nodes 20 --seed 2018 \
#--target gtsrb_binary_mlpbnn_approx --round 100
#
#python train_binarynn.py --dataset gtsrb_binary --n_classes 2 --hidden-nodes 20 --seed 2018 \
#--target gtsrb_binary_mlpbnn_approx_ep1 --round 100 --normal-noise --epsilon 0.1
#
#python train_binarynn.py --dataset gtsrb_binary --n_classes 2 --hidden-nodes 20 --seed 2018 \
#--target gtsrb_binary_mlpbnn_approx_ep004 --round 100 --normal-noise --epsilon 0.004
#
#python train_binarynn.py --dataset gtsrb_binary --n_classes 2 --hidden-nodes 20 --seed 2018 \
#--target gtsrb_binary_mlpbnn_approx_ep2 --round 100 --normal-noise --epsilon 0.2

#version=mlp6
#
#
#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 1 \
#--target gtsrb_binary_1_nr075_${version}_sign_01loss_2000_0_w1_h1.pkl --dataset gtsrb_binary  --seed 0  \
#--version $version --act sign --width 1
#
#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 1 \
#--target gtsrb_binary_1_nr075_${version}_sign_01loss_2000_0_w10_h1.pkl --dataset gtsrb_binary  --seed 0  \
#--version $version --act sign --width 10
#
#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 120 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 20 \
#--target gtsrb_binary_1_nr075_${version}_sign_01loss_120_0_w1_h20.pkl --dataset gtsrb_binary  --seed 0  \
#--version $version --act sign --width 1