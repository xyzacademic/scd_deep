#!/bin/bash -l

#SBATCH -p datasci
#SBATCH --job-name=mnist_job
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

#

cd ..

gpu=0

#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 100 --b-ratio 0.2 \
#--updated-features 64 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target mnist_hinge_01.pkl --dataset mnist  --seed 2018 --c 0.1
#
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 2 \
#--train-size 200 --target mnist_hinge_01 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
#done


#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 100 --b-ratio 0.2 \
#--updated-features 64 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target mnist_hinge_02.pkl --dataset mnist  --seed 2018 --c 0.2
#
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 2 \
#--train-size 200 --target mnist_hinge_02 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
#done
#
#
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 100 --b-ratio 0.2 \
#--updated-features 64 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target mnist_hinge_03.pkl --dataset mnist  --seed 2018 --c 0.3
#
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 2 \
#--train-size 200 --target mnist_hinge_03 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
#done

#
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 100 --b-ratio 0.2 \
#--updated-features 64 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target mnist_hinge_04.pkl --dataset mnist  --seed 2018 --c 0.4
#
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 2 \
#--train-size 200 --target mnist_hinge_04 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
#done
#
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 100 --b-ratio 0.2 \
#--updated-features 64 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target mnist_hinge_05.pkl --dataset mnist  --seed 2018 --c 0.5
#
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 2 \
#--train-size 200 --target mnist_hinge_05 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
#done

python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 100 --b-ratio 0.2 \
--updated-features 64 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
--target mnist_hinge_06.pkl --dataset mnist  --seed 2018 --c 0.6

for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
do
python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 2 \
--train-size 200 --target mnist_hinge_06 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
done

#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 100 --b-ratio 0.2 \
#--updated-features 64 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target mnist_hinge_07.pkl --dataset mnist  --seed 2018 --c 0.7
##
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 2 \
#--train-size 200 --target mnist_hinge_07 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
#done
#
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 100 --b-ratio 0.2 \
#--updated-features 64 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target mnist_hinge_08.pkl --dataset mnist  --seed 2018 --c 0.8
#
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 2 \
#--train-size 200 --target mnist_hinge_08 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
#done
#
#
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 100 --b-ratio 0.2 \
#--updated-features 64 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target mnist_hinge_09.pkl --dataset mnist  --seed 2018 --c 0.9
#
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 2 \
#--train-size 200 --target mnist_hinge_09 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
#done
#
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 100 --b-ratio 0.2 \
#--updated-features 64 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target mnist_hinge_10.pkl --dataset mnist  --seed 2018 --c 1.0
#
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 2 \
#--train-size 200 --target mnist_hinge_10 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024
#done

