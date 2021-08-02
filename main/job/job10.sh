#!/bin/sh

#SBATCH -p datasci


#SBATCH --job-name=cifar10_job
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

cd ..
python train_cnn01_ce.py --nrows 1506 --localit 1 --updated_conv_features 3 --updated_fc_nodes 1 --updated_conv_nodes 1 --width 100 --normalize 1 --percentile 1 --fail_count 1 --loss bce --fc_diversity 0 --init uniform --no_bias 0 --scale 1 --w-inc1 0.1 --version toy --seed 0 --iters 1000 --dataset cifar10 --n_classes 10  --target cifar10_toy_softmax.pkl --updated_fc_ratio 1 --updated_conv_ratio 1 --verbose

#gpu=0
#
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.015625 --Lambda 0.05 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_binary_toy2_i3_bce_div_s2_16 --random-sign 1 --seed $seed --dataset cifar10_binary --oracle-size 1024 --n_classes 2
#python bb_attack.py --epsilon 0.015625 --Lambda 0.05 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_binary_resnet18 --random-sign 1 --seed $seed --dataset cifar10_binary --oracle-size 1024 --n_classes 2
#python bb_attack.py --epsilon 0.015625 --Lambda 0.05 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_binary_toy2_s2_bp --random-sign 1 --seed $seed --dataset cifar10_binary --oracle-size 1024 --n_classes 2
#python bb_attack.py --epsilon 0.031250 --Lambda 0.05 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_binary_toy2_i3_bce_div_s2_16 --random-sign 1 --seed $seed --dataset cifar10_binary --oracle-size 1024 --n_classes 2
#python bb_attack.py --epsilon 0.031250 --Lambda 0.05 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_binary_resnet18 --random-sign 1 --seed $seed --dataset cifar10_binary --oracle-size 1024 --n_classes 2
#python bb_attack.py --epsilon 0.031250 --Lambda 0.05 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_binary_toy2_s2_bp --random-sign 1 --seed $seed --dataset cifar10_binary --oracle-size 1024 --n_classes 2
#python bb_attack.py --epsilon 0.062500 --Lambda 0.05 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_binary_toy2_i3_bce_div_s2_16 --random-sign 1 --seed $seed --dataset cifar10_binary --oracle-size 1024 --n_classes 2
#python bb_attack.py --epsilon 0.062500 --Lambda 0.05 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_binary_resnet18 --random-sign 1 --seed $seed --dataset cifar10_binary --oracle-size 1024 --n_classes 2
#python bb_attack.py --epsilon 0.062500 --Lambda 0.05 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_binary_toy2_s2_bp --random-sign 1 --seed $seed --dataset cifar10_binary --oracle-size 1024 --n_classes 2
#python bb_attack.py --epsilon 0.125000 --Lambda 0.05 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_binary_toy2_i3_bce_div_s2_16 --random-sign 1 --seed $seed --dataset cifar10_binary --oracle-size 1024 --n_classes 2
#python bb_attack.py --epsilon 0.125000 --Lambda 0.05 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_binary_resnet18 --random-sign 1 --seed $seed --dataset cifar10_binary --oracle-size 1024 --n_classes 2
#python bb_attack.py --epsilon 0.125000 --Lambda 0.05 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_binary_toy2_s2_bp --random-sign 1 --seed $seed --dataset cifar10_binary --oracle-size 1024 --n_classes 2
#python bb_attack.py --epsilon 0.250000 --Lambda 0.05 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_binary_toy2_i3_bce_div_s2_16 --random-sign 1 --seed $seed --dataset cifar10_binary --oracle-size 1024 --n_classes 2
#python bb_attack.py --epsilon 0.250000 --Lambda 0.05 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_binary_resnet18 --random-sign 1 --seed $seed --dataset cifar10_binary --oracle-size 1024 --n_classes 2
#python bb_attack.py --epsilon 0.250000 --Lambda 0.05 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_binary_toy2_s2_bp --random-sign 1 --seed $seed --dataset cifar10_binary --oracle-size 1024 --n_classes 2
#
#done
