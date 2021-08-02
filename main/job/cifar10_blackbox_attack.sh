#!/bin/sh

#SBATCH -p datasci4

#SBATCH --job-name=cifar10_bbattack
#SBATCH --gres=gpu:1
#SBATCH --mem=16G


cd ..

gpu=0

for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
do


python bb_attack_rdcnn.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target cifar10_simclr_svm --random-sign 1 --seed $seed --dataset cifar10 \
--oracle-size 256 --n_classes 2

#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_32_nr075_mlp1_sign_01loss_1000_w10_h1 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2

#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_32_nr075_mlp2_sign_01loss_1000_w10_h1 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2
##
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_32_nr075_mlp3_sign_01loss_1000_w10_h1 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_32_nr075_mlp4_sign_01loss_1000_w10_h1 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2

#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_mlp_01_reluact_1000_1 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2

#python bb_attack_bnn.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_mlp1_bnn_ste_sign.h5 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2

#python bb_attack_bnn.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_mlp1_bnn_approx_sign.h5 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2
#
#python bb_attack_bnn.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_mlp1_bnn_swish_sign.h5 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2

#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_12_mlp_01_reluact_1000 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2
##
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_12_mlp_01_01act_1000 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2



#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_scd01mlp_32_br02_nr075_ni1000_i1 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2

#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_scd01_32_br02_nr075_ni1000_i1_s8 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2 --binarize --eps 8
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_scd01mlp_32_br02_nr075_ni1000_i1_s8 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2 --binarize --eps 8
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_scd01_32_br02_nr075_ni1000_i1_s128 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2 --binarize --eps 128
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_scd01mlp_32_br02_nr075_ni1000_i1_s128 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2 --binarize --eps 128

#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_scd01_32_br02_nr075_ni1000_i1_sinf --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2 --binarize

#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_scd01mlp_32_br02_nr075_ni8000_i1_h20 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_scd01mlp_32_br02_nr075_ni8000_i1_h40 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_scd01mlp_32_br02_nr075_ni8000_i1_h80 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_mlp80 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_mlp20_32_soft --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_mlp80_32_soft --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_mlp400_32_soft --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10

#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_mlp20 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_mlp400 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10

#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_mlp80 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10


#python bb_attack_.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_lenet --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10
#
#python bb_attack_.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_simplenet20 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10
#
#python bb_attack_.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_simplenet80 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10

#python bb_attack_.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 50 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_simplenet20_32 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10
#
#python bb_attack_.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 50 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_simplenet80_32 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10


#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_spm80 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_spm40 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10

#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_scd01mlp_1_br02_nr075_ni8000_i1_h20 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_scd01mlp_1_br02_nr075_ni8000_i1_h40 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_scd01mlp_1_br02_nr075_ni8000_i1_h80 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_scd01mlp_1_br02_nr075_ni8000_i1_h400 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10

#python bb_attack_.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_cnn --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_scd01mlp_32_br02_nr075_ni1000_i1_sinf --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2 --binarize
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_scd01mlp_32_br05_nr075_ni500_i1 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10

#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_scd01mlp_32_br02_nr075_ni1000_i1 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_scd01mlp_32_br05_nr075_ni1000 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10


#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_svm --random-sign 1 --seed $seed --dataset cifar10 --oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_mlp --random-sign 1 --seed $seed --dataset cifar10 --oracle-size 1024 --n_classes 10
#
#
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_mlp1vall --random-sign 1 --seed $seed --dataset cifar10 --oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_svm1vall_32vote --random-sign 1 --seed $seed --dataset cifar10 --oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_mlp1vall_32vote --random-sign 1 --seed $seed --dataset cifar10 --oracle-size 1024 --n_classes 10

#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_dt --random-sign 1 --seed $seed --dataset cifar10 --oracle-size 1024 --n_classes 2
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_rf --random-sign 1 --seed $seed --dataset cifar10 --oracle-size 1024 --n_classes 2
#

#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_svm --random-sign 1 --seed $seed --dataset cifar10 --oracle-size 1024 --n_classes 2
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_mlp --random-sign 1 --seed $seed --dataset cifar10 --oracle-size 1024 --n_classes 2

#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_svm_s8 --random-sign 1 --seed $seed --dataset cifar10 --oracle-size 1024 --n_classes 2 --binarize --eps 8
##
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_mlp_s8 --random-sign 1 --seed $seed --dataset cifar10 --oracle-size 1024 --n_classes 2 --binarize --eps 8
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_svm_s128 --random-sign 1 --seed $seed --dataset cifar10 --oracle-size 1024 --n_classes 2 --binarize --eps 128
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_mlp_s128 --random-sign 1 --seed $seed --dataset cifar10 --oracle-size 1024 --n_classes 2 --binarize --eps 128


#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_svm_sinf --random-sign 1 --seed $seed --dataset cifar10 --oracle-size 1024 --n_classes 2 --binarize --eps 128
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_mlp_sinf --random-sign 1 --seed $seed --dataset cifar10 --oracle-size 1024 --n_classes 2 --binarize --eps 128

done
