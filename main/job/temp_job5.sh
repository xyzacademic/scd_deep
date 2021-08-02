#!/bin/bash -l
#SBATCH -p datasci
#SBATCH --job-name=cifar10_job
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
cd ..
gpu=0

python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.05 --num-iters 1000 --b-ratio 0.2         --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1         --target cifar10_hinge_12.pkl --dataset cifar10  --seed 2018 --c 1.2
for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
do
python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001         --train-size 200 --target cifar10_hinge_12 --random-sign 1 --seed $seed --dataset cifar10         --oracle-size 1024 --n_classes 2
done
