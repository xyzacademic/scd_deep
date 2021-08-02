#!/bin/sh

#SBATCH -p datasci4


#SBATCH --job-name=cifar10_train
#SBATCH --gres=gpu:4
#SBATCH --mem=32G

cd ..

gpu=0

for seed in 2018
do

python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 2 \
--train-size 200 --target mnist_scd01_1_br02_nr075_ni1000_i1 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
#
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 2 \
#--train-size 200 --target mnist_scd01_32_br02_nr075_ni500_i1_ep9 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
#
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 2 \
#--train-size 200 --target mnist_scd01_32_br02_nr075_ni500_i1_ep10 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
#
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 2 \
#--train-size 200 --target mnist_scd01_32_br02_nr075_ni500_i1_ep15 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
#
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 2 \
#--train-size 200 --target mnist_scd01_32_br02_nr075_ni500_i1_ep20 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
#
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 2 \
#--train-size 200 --target mnist_scd01mlp_32_br02_nr075_ni1000_i1 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024

#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 2 \
#--train-size 200 --target mnist_scd01mlp_32_br02_nr075_ni1000_i1_ep9 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
#
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 2 \
#--train-size 200 --target mnist_scd01mlp_32_br02_nr075_ni1000_i1_ep10 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
#
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 2 \
#--train-size 200 --target mnist_scd01mlp_32_br02_nr075_ni1000_i1_ep15 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024

#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 2 \
#--train-size 200 --target mnist_scd01mlp_32_br02_nr075_ni1000_i1_ep20 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024


#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001  \
#--train-size 200 --target mnist_scd01mlp_32_br02_nr075_ni500_i1 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 

#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001  \
#--train-size 200 --target mnist_scd01mlp_32_br05_nr075_ni500_i1 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 
#
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001  \
#--train-size 200 --target mnist_scd01mlp_32_br02_nr075_ni1000_i1 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 
#
#
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001  --n_classes 2 \
#--train-size 200 --target mnist_svm --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
#
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001  --n_classes 2 \
#--train-size 200 --target mnist_svm_ep1 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
#
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001  --n_classes 2 \
#--train-size 200 --target mnist_svm_ep2 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
#
#
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001  --n_classes 2 \
#--train-size 200 --target mnist_svm_ep3 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
#
#
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001  --n_classes 2 \
#--train-size 200 --target mnist_svm_ep4 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024


#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001  --n_classes 2  \
#--train-size 200 --target mnist_mlp --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
#
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001  --n_classes 2  \
#--train-size 200 --target mnist_mlp_ep1 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
#
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001  --n_classes 2  \
#--train-size 200 --target mnist_mlp_ep2 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
#
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001  --n_classes 2  \
#--train-size 200 --target mnist_mlp_ep3 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
#
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001  --n_classes 2  \
#--train-size 200 --target mnist_mlp_ep4 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
#
#
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001  \
#--train-size 200 --target mnist_mlp1vall --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 

#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001  \
#--train-size 200 --target mnist_svm1vall_32vote --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 
#
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001  \
#--train-size 200 --target mnist_mlp1vall_32vote --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 

#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001  \
#--train-size 200 --target mnist_dt --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 
#
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001  \
#--train-size 200 --target mnist_rf --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 



python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 \
--train-size 200 --target mnist_svm --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 --n_classes 2

python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 \
--train-size 200 --target mnist_mlp --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 --n_classes 2

python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 \
--train-size 200 --target mnist_svm_s8 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 --n_classes 2 --binarize --eps 8
#
python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 \
--train-size 200 --target mnist_mlp_s8 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 --n_classes 2 --binarize --eps 8

python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 \
--train-size 200 --target mnist_svm_s128 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 --n_classes 2 --binarize --eps 128

python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 \
--train-size 200 --target mnist_mlp_s128 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 --n_classes 2 --binarize --eps 128

python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 \
--train-size 200 --target mnist_scd01_32_br02_nr075_ni1000_i1_s8 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 --n_classes 2 --binarize --eps 8
#
python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 \
--train-size 200 --target mnist_scd01mlp_32_br02_nr075_ni1000_i1_s8 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 --n_classes 2 --binarize --eps 8

python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 \
--train-size 200 --target mnist_scd01_32_br02_nr075_ni1000_i1_s128 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 --n_classes 2 --binarize --eps 128

python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 \
--train-size 200 --target mnist_scd01mlp_32_br02_nr075_ni1000_i1_s128 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 --n_classes 2 --binarize --eps 128

done
