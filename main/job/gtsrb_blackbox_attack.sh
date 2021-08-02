#!/bin/bash -l

#SBATCH -p datasci3

#SBATCH --job-name=gtsrb_scd_attack
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
cd ..
gpu=1
seed=2018
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target gtsrb_scd01_32_br02_nr075_ni500_i1 --random-sign 1 --seed $seed --dataset gtsrb \
#--oracle-size 1024 --n_classes 2
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target gtsrb_scd01_32_br02_nr075_ni500_i1_ep1 --random-sign 1 --seed $seed --dataset gtsrb \
#--oracle-size 1024 --n_classes 2
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target gtsrb_scd01_32_br02_nr075_ni500_i1_ep2 --random-sign 1 --seed $seed --dataset gtsrb \
#--oracle-size 1024 --n_classes 2
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target gtsrb_scd01_32_br02_nr075_ni500_i1_ep3 --random-sign 1 --seed $seed --dataset gtsrb \
#--oracle-size 1024 --n_classes 2

#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target gtsrb_scd01_32_br02_nr075_ni500_i1_ep4 --random-sign 1 --seed $seed --dataset gtsrb \
#--oracle-size 1024 --n_classes 2

#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target gtsrb_scd01mlp_32_br02_nr075_ni500_i1 --random-sign 1 --seed $seed --dataset gtsrb \
#--oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target gtsrb_scd01mlp_32_br05_nr075_ni500_i1 --random-sign 1 --seed $seed --dataset gtsrb \
#--oracle-size 1024 --n_classes 10

#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target gtsrb_scd01mlp_32_br02_nr075_ni1000_i1 --random-sign 1 --seed $seed --dataset gtsrb \
#--oracle-size 1024 --n_classes 2
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target gtsrb_scd01mlp_32_br02_nr075_ni1000_i1_ep1 --random-sign 1 --seed $seed --dataset gtsrb \
#--oracle-size 1024 --n_classes 2
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target gtsrb_scd01mlp_32_br02_nr075_ni1000_i1_ep2 --random-sign 1 --seed $seed --dataset gtsrb \
#--oracle-size 1024 --n_classes 2
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target gtsrb_scd01mlp_32_br02_nr075_ni1000_i1_ep3 --random-sign 1 --seed $seed --dataset gtsrb \
#--oracle-size 1024 --n_classes 2

#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 30 --lr 0.0001 \
#--train-size 200 --target gtsrb_scd01mlp_32_br02_nr075_ni1000_i1 --random-sign 1 --seed $seed --dataset gtsrb \
#--oracle-size 1024 --n_classes 2
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target gtsrb_scd01mlp_32_br05_nr075_ni1000 --random-sign 1 --seed $seed --dataset gtsrb \
#--oracle-size 1024 --n_classes 10


#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target gtsrb_svm --random-sign 1 --seed $seed --dataset gtsrb --oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target gtsrb_mlp --random-sign 1 --seed $seed --dataset gtsrb --oracle-size 1024 --n_classes 10

#
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target gtsrb_mlp1vall --random-sign 1 --seed $seed --dataset gtsrb --oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target gtsrb_svm1vall_32vote --random-sign 1 --seed $seed --dataset gtsrb --oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target gtsrb_mlp1vall_32vote --random-sign 1 --seed $seed --dataset gtsrb --oracle-size 1024 --n_classes 10

#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target gtsrb_dt --random-sign 1 --seed $seed --dataset gtsrb --oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target gtsrb_rf --random-sign 1 --seed $seed --dataset gtsrb --oracle-size 1024 --n_classes 10


python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target gtsrb_binary_svm --random-sign 1 --seed $seed --dataset gtsrb_binary --oracle-size 1024 --n_classes 2

python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target gtsrb_binary_mlp --random-sign 1 --seed $seed --dataset gtsrb_binary --oracle-size 1024 --n_classes 2

python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target gtsrb_binary_svm_s8 --random-sign 1 --seed $seed --dataset gtsrb_binary --oracle-size 1024 --n_classes 2 --binarize --eps 8
#
python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target gtsrb_binary_mlp_s8 --random-sign 1 --seed $seed --dataset gtsrb_binary --oracle-size 1024 --n_classes 2 --binarize --eps 8

python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target gtsrb_binary_svm_s128 --random-sign 1 --seed $seed --dataset gtsrb_binary --oracle-size 1024 --n_classes 2 --binarize --eps 128

python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target gtsrb_binary_mlp_s128 --random-sign 1 --seed $seed --dataset gtsrb_binary --oracle-size 1024 --n_classes 2 --binarize --eps 128


python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target gtsrb_binary_scd01_32_br02_nr075_ni1000_i1 --random-sign 1 --seed $seed --dataset gtsrb_binary --oracle-size 1024 --n_classes 2

python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target gtsrb_binary_scd01mlp_32_br02_nr075_ni1000_i1 --random-sign 1 --seed $seed --dataset gtsrb_binary --oracle-size 1024 --n_classes 2

python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target gtsrb_binary_scd01_32_br02_nr075_ni1000_i1_s8 --random-sign 1 --seed $seed --dataset gtsrb_binary --oracle-size 1024 --n_classes 2 --binarize --eps 8
#
python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target gtsrb_binary_scd01mlp_32_br02_nr075_ni1000_i1_s8 --random-sign 1 --seed $seed --dataset gtsrb_binary --oracle-size 1024 --n_classes 2 --binarize --eps 8

python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target gtsrb_binary_scd01_32_br02_nr075_ni1000_i1_s128 --random-sign 1 --seed $seed --dataset gtsrb_binary --oracle-size 1024 --n_classes 2 --binarize --eps 128

python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target gtsrb_binary_scd01mlp_32_br02_nr075_ni1000_i1_s128 --random-sign 1 --seed $seed --dataset gtsrb_binary --oracle-size 1024 --n_classes 2 --binarize --eps 128
