#!/bin/sh

c0=0
c1=1

python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 --votes 8 \
--no_bias 0 --scale 1 --cnn 0 --version mlp01scale --act sign \
--c0 ${c0} --c1 ${c1} --target cifar10_binary_${c0}${c1}_eps0.0625_mlp01scale_sign_i1_01loss_b7500_lrc0.05_lrf0.17_nb0_nw1_dm0_upc1_upf1_ucf32_normal

python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 --votes 8 \
--no_bias 0 --scale 1 --cnn 0 --version mlp01scale --act sign \
--c0 ${c0} --c1 ${c1} --target cifar10_binary_${c0}${c1}_eps0.125_mlp01scale_sign_i1_01loss_b7500_lrc0.05_lrf0.17_nb0_nw1_dm0_upc1_upf1_ucf32_normal

python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 --votes 8 \
--no_bias 0 --scale 1 --cnn 0 --version mlp01scale --act sign \
--c0 ${c0} --c1 ${c1} --target cifar10_binary_${c0}${c1}_eps0.25_mlp01scale_sign_i1_01loss_b7500_lrc0.05_lrf0.17_nb0_nw1_dm0_upc1_upf1_ucf32_normal

python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 --votes 8 \
--no_bias 0 --scale 1 --cnn 0 --version mlp01scale --act sign \
--c0 ${c0} --c1 ${c1} --target cifar10_binary_${c0}${c1}_eps0.5_mlp01scale_sign_i1_01loss_b7500_lrc0.05_lrf0.17_nb0_nw1_dm0_upc1_upf1_ucf32_normal

python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 --votes 8 \
--no_bias 0 --scale 1 --cnn 0 --version mlp01scale --act sign \
--c0 ${c0} --c1 ${c1} --target cifar10_binary_${c0}${c1}_eps1.0_mlp01scale_sign_i1_01loss_b7500_lrc0.05_lrf0.17_nb0_nw1_dm0_upc1_upf1_ucf32_normal


c0=2
c1=3

python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 --votes 8 \
--no_bias 0 --scale 1 --cnn 0 --version mlp01scale --act sign \
--c0 ${c0} --c1 ${c1} --target cifar10_binary_${c0}${c1}_eps0.0625_mlp01scale_sign_i1_01loss_b7500_lrc0.05_lrf0.17_nb0_nw1_dm0_upc1_upf1_ucf32_normal

python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 --votes 8 \
--no_bias 0 --scale 1 --cnn 0 --version mlp01scale --act sign \
--c0 ${c0} --c1 ${c1} --target cifar10_binary_${c0}${c1}_eps0.125_mlp01scale_sign_i1_01loss_b7500_lrc0.05_lrf0.17_nb0_nw1_dm0_upc1_upf1_ucf32_normal

python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 --votes 8 \
--no_bias 0 --scale 1 --cnn 0 --version mlp01scale --act sign \
--c0 ${c0} --c1 ${c1} --target cifar10_binary_${c0}${c1}_eps0.25_mlp01scale_sign_i1_01loss_b7500_lrc0.05_lrf0.17_nb0_nw1_dm0_upc1_upf1_ucf32_normal

python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 --votes 8 \
--no_bias 0 --scale 1 --cnn 0 --version mlp01scale --act sign \
--c0 ${c0} --c1 ${c1} --target cifar10_binary_${c0}${c1}_eps0.5_mlp01scale_sign_i1_01loss_b7500_lrc0.05_lrf0.17_nb0_nw1_dm0_upc1_upf1_ucf32_normal

python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 --votes 8 \
--no_bias 0 --scale 1 --cnn 0 --version mlp01scale --act sign \
--c0 ${c0} --c1 ${c1} --target cifar10_binary_${c0}${c1}_eps1.0_mlp01scale_sign_i1_01loss_b7500_lrc0.05_lrf0.17_nb0_nw1_dm0_upc1_upf1_ucf32_normal

c0=4
c1=5

python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 --votes 8 \
--no_bias 0 --scale 1 --cnn 0 --version mlp01scale --act sign \
--c0 ${c0} --c1 ${c1} --target cifar10_binary_${c0}${c1}_eps0.0625_mlp01scale_sign_i1_01loss_b7500_lrc0.05_lrf0.17_nb0_nw1_dm0_upc1_upf1_ucf32_normal

python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 --votes 8 \
--no_bias 0 --scale 1 --cnn 0 --version mlp01scale --act sign \
--c0 ${c0} --c1 ${c1} --target cifar10_binary_${c0}${c1}_eps0.125_mlp01scale_sign_i1_01loss_b7500_lrc0.05_lrf0.17_nb0_nw1_dm0_upc1_upf1_ucf32_normal

python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 --votes 8 \
--no_bias 0 --scale 1 --cnn 0 --version mlp01scale --act sign \
--c0 ${c0} --c1 ${c1} --target cifar10_binary_${c0}${c1}_eps0.25_mlp01scale_sign_i1_01loss_b7500_lrc0.05_lrf0.17_nb0_nw1_dm0_upc1_upf1_ucf32_normal

python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 --votes 8 \
--no_bias 0 --scale 1 --cnn 0 --version mlp01scale --act sign \
--c0 ${c0} --c1 ${c1} --target cifar10_binary_${c0}${c1}_eps0.5_mlp01scale_sign_i1_01loss_b7500_lrc0.05_lrf0.17_nb0_nw1_dm0_upc1_upf1_ucf32_normal

python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 --votes 8 \
--no_bias 0 --scale 1 --cnn 0 --version mlp01scale --act sign \
--c0 ${c0} --c1 ${c1} --target cifar10_binary_${c0}${c1}_eps1.0_mlp01scale_sign_i1_01loss_b7500_lrc0.05_lrf0.17_nb0_nw1_dm0_upc1_upf1_ucf32_normal

c0=6
c1=7

python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 --votes 8 \
--no_bias 0 --scale 1 --cnn 0 --version mlp01scale --act sign \
--c0 ${c0} --c1 ${c1} --target cifar10_binary_${c0}${c1}_eps0.0625_mlp01scale_sign_i1_01loss_b7500_lrc0.05_lrf0.17_nb0_nw1_dm0_upc1_upf1_ucf32_normal

python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 --votes 8 \
--no_bias 0 --scale 1 --cnn 0 --version mlp01scale --act sign \
--c0 ${c0} --c1 ${c1} --target cifar10_binary_${c0}${c1}_eps0.125_mlp01scale_sign_i1_01loss_b7500_lrc0.05_lrf0.17_nb0_nw1_dm0_upc1_upf1_ucf32_normal

python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 --votes 8 \
--no_bias 0 --scale 1 --cnn 0 --version mlp01scale --act sign \
--c0 ${c0} --c1 ${c1} --target cifar10_binary_${c0}${c1}_eps0.25_mlp01scale_sign_i1_01loss_b7500_lrc0.05_lrf0.17_nb0_nw1_dm0_upc1_upf1_ucf32_normal

python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 --votes 8 \
--no_bias 0 --scale 1 --cnn 0 --version mlp01scale --act sign \
--c0 ${c0} --c1 ${c1} --target cifar10_binary_${c0}${c1}_eps0.5_mlp01scale_sign_i1_01loss_b7500_lrc0.05_lrf0.17_nb0_nw1_dm0_upc1_upf1_ucf32_normal

python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 --votes 8 \
--no_bias 0 --scale 1 --cnn 0 --version mlp01scale --act sign \
--c0 ${c0} --c1 ${c1} --target cifar10_binary_${c0}${c1}_eps1.0_mlp01scale_sign_i1_01loss_b7500_lrc0.05_lrf0.17_nb0_nw1_dm0_upc1_upf1_ucf32_normal

c0=8
c1=9

python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 --votes 8 \
--no_bias 0 --scale 1 --cnn 0 --version mlp01scale --act sign \
--c0 ${c0} --c1 ${c1} --target cifar10_binary_${c0}${c1}_eps0.0625_mlp01scale_sign_i1_01loss_b7500_lrc0.05_lrf0.17_nb0_nw1_dm0_upc1_upf1_ucf32_normal

python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 --votes 8 \
--no_bias 0 --scale 1 --cnn 0 --version mlp01scale --act sign \
--c0 ${c0} --c1 ${c1} --target cifar10_binary_${c0}${c1}_eps0.125_mlp01scale_sign_i1_01loss_b7500_lrc0.05_lrf0.17_nb0_nw1_dm0_upc1_upf1_ucf32_normal

python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 --votes 8 \
--no_bias 0 --scale 1 --cnn 0 --version mlp01scale --act sign \
--c0 ${c0} --c1 ${c1} --target cifar10_binary_${c0}${c1}_eps0.25_mlp01scale_sign_i1_01loss_b7500_lrc0.05_lrf0.17_nb0_nw1_dm0_upc1_upf1_ucf32_normal

python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 --votes 8 \
--no_bias 0 --scale 1 --cnn 0 --version mlp01scale --act sign \
--c0 ${c0} --c1 ${c1} --target cifar10_binary_${c0}${c1}_eps0.5_mlp01scale_sign_i1_01loss_b7500_lrc0.05_lrf0.17_nb0_nw1_dm0_upc1_upf1_ucf32_normal

python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 --votes 8 \
--no_bias 0 --scale 1 --cnn 0 --version mlp01scale --act sign \
--c0 ${c0} --c1 ${c1} --target cifar10_binary_${c0}${c1}_eps1.0_mlp01scale_sign_i1_01loss_b7500_lrc0.05_lrf0.17_nb0_nw1_dm0_upc1_upf1_ucf32_normal

