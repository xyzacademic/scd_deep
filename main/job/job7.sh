#!/bin/bash -l

#SBATCH -p datasci


#SBATCH --job-name=all_data
#SBATCH --mem=32G
cd ..

gpu=1
seed=2019
#python train_mlp_1vall.py --dataset cifar10 --hidden-nodes 20 --n_classes 10 --lr 0.01 --iters 200 --round 32 --target cifar10_mlp1vall_32vote.pkl --save
#python train_svm_1vall.py --dataset cifar10 --hidden-nodes 20 --n_classes 10 --c 0.01 --round 32 --target cifar10_svm1vall_32vote.pkl  --save
#python train_mlp_1vall.py --dataset mnist --hidden-nodes 20 --n_classes 10 --lr 0.01 --iters 200 --round 32 --target mnist_mlp1vall_32vote.pkl  --save
#python train_svm_1vall.py --dataset mnist --hidden-nodes 20 --n_classes 10 --c 0.01 --round 32 --target mnist_svm1vall_32vote.pkl  --save


#python train_mlp_1vall.py --dataset imagenet --hidden-nodes 20 --n_classes 10 --lr 0.001 --iters 200 --target imagenet_mlp1vall_20.pkl
#python train_mlp_1vall.py --dataset imagenet --hidden-nodes 400 --n_classes 10 --lr 0.001 --iters 200 --target imagenet_mlp1vall_400.pkl
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_mlp1vall_32vote --random-sign 1 --seed $seed --dataset cifar10 --oracle-size 1024 --n_classes 10

#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_svm1vall_32vote --random-sign 1 --seed $seed --dataset cifar10 --oracle-size 1024 --n_classes 10


#python bb_attack.py --epsilon 0.2 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 10 \
#--train-size 200 --target mnist_mlp1vall_32vote --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 --n_classes 10

#python bb_attack.py --epsilon 0.2 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 10 \
#--train-size 200 --target mnist_svm1vall_32vote --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target imagenet_mlp1vall_20 --random-sign 1 --seed $seed --dataset imagenet --oracle-size 256 --n_classes 10

#python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target imagenet_mlp1vall_400 --random-sign 1 --seed $seed --dataset imagenet --oracle-size 256 --n_classes 10

#python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target imagenet_scd01mlp_32_br02_nr075_ni1000_i1 --random-sign 1 --seed $seed --dataset imagenet --oracle-size 256 --n_classes 10

#python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001  --n_classes 10 \
#--train-size 200 --target imagenet_scd01_32_br02_nr075_ni500_i1 --random-sign 1 --seed $seed --dataset imagenet --oracle-size 256


#python train_dt.py --dataset mnist --n_classes 10 --target mnist_rf_32vote.pkl --save
#python train_dt.py --dataset cifar10 --n_classes 10 --target cifar10_rf_32vote.pkl --save
#python train_dt.py --dataset imagenet --n_classes 10 --target imagenet_rf_32vote.pkl --save

#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_rf_32vote --random-sign 1 --seed $seed --dataset cifar10 --oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.2 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 10 \
#--train-size 200 --target mnist_rf_32vote --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 --n_classes 10

#python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target imagenet_rf_32vote --random-sign 1 --seed $seed --dataset imagenet --oracle-size 256 --n_classes 10


#python train_svm.py --dataset mnist --n_classes 10 --c 0.01 --target mnist_svm.pkl
#python train_mlp.py --dataset mnist --n_classes 10 --hidden-nodes 20 --lr 0.01 --iters 200 --target mnist_mlp.pkl
#
python train_svm.py --dataset cifar10 --n_classes 2 --c 0.01 --target cifar10_svm.pkl --save --seed 2018
python train_mlp.py --dataset cifar10 --n_classes 2 --hidden-nodes 20 --lr 0.01 --iters 200 --target cifar10_mlp.pkl --save --seed 2018

python train_svm.py --dataset cifar10 --n_classes 2 --c 0.01 --target cifar10_svm_s128.pkl --save  --binarize --eps 128 --seed 2018
python train_mlp.py --dataset cifar10 --n_classes 2 --hidden-nodes 20 --lr 0.01 --iters 200 --target cifar10_mlp_s128.pkl --save  --binarize --eps 128 --seed 2018
python train_svm.py --dataset cifar10 --n_classes 2 --c 0.01 --target cifar10_svm_s8.pkl --save  --binarize --eps 8 --seed 2018
python train_mlp.py --dataset cifar10 --n_classes 2 --hidden-nodes 20 --lr 0.01 --iters 200 --target cifar10_mlp_s8.pkl --save  --binarize --eps 8 --seed 2018

python train_svm.py --dataset cifar10 --n_classes 2 --c 0.01 --target cifar10_svm_sinf.pkl --save  --binarize --eps 8 --seed 2018
python train_mlp.py --dataset cifar10 --n_classes 2 --hidden-nodes 20 --lr 0.01 --iters 200 --target cifar10_mlp_sinf.pkl --save  --binarize --eps 8 --seed 2018

#
#python train_svm.py --dataset imagenet --n_classes 10 --c 0.001 --target imagenet_svm.pkl
#python train_mlp.py --dataset imagenet --n_classes 10 --hidden-nodes 400 --lr 0.01 --iters 200 --target imagenet_mlp.pkl


#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_scd01_32_br02_nr075_ni500_i1 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_scd01mlp_32_br02_nr075_ni1000_i1 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.2 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 10 \
#--train-size 200 --target mnist_scd01_32_br02_nr075_ni500_i1 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.2 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 10 \
#--train-size 200 --target mnist_scd01mlp_32_br02_nr075_ni1000_i1 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 --n_classes 10


#python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001  --n_classes 10 \
#--train-size 200 --target imagenet_scd01mlp_32_br02_nr075_ni500_i1 --random-sign 1 --seed $seed --dataset imagenet --oracle-size 256
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001  --n_classes 10 \
#--train-size 200 --target imagenet_scd01mlp_32_br02_nr075_ni1000_i1 --random-sign 1 --seed $seed --dataset imagenet --oracle-size 256
#


#python train_mlp_1vall.py --dataset gtsrb --hidden-nodes 20 --n_classes 10 --lr 0.01 --iters 200 --round 32 --target gtsrb_mlp1vall_32vote.pkl --save
#python train_svm_1vall.py --dataset gtsrb --n_classes 10 --c 0.01 --round 32 --target gtsrb_svm1vall_32vote.pkl  --save

python train_svm.py --dataset gtsrb --n_classes 10 --c 0.01  --target gtsrb_svm.pkl  --save
python train_mlp.py --dataset gtsrb --hidden-nodes 20 --n_classes 10 --lr 0.001 --iters 2000 --target gtsrb_mlp.pkl --save


python train_svm.py --dataset cifar10 --n_classes 2 --c 0.01 --target cifar10_svm.pkl --save --seed 2018
python train_mlp.py --dataset cifar10 --n_classes 2 --hidden-nodes 20 --lr 0.01 --iters 200 --target cifar10_mlp.pkl --save --seed 2018

python train_svm.py --dataset mnist --n_classes 2 --c 0.01 --target mnist_svm_s128.pkl --save  --binarize --eps 128 --seed 2018
python train_mlp.py --dataset mnist --n_classes 2 --hidden-nodes 20 --lr 0.01 --iters 200 --target mnist_mlp_s128.pkl --save  --binarize --eps 128 --seed 2018
python train_svm.py --dataset mnist --n_classes 2 --c 0.01 --target mnist_svm_s8.pkl --save  --binarize --eps 8 --seed 2018
python train_mlp.py --dataset mnist --n_classes 2 --hidden-nodes 20 --lr 0.01 --iters 200 --target mnist_mlp_s8.pkl --save  --binarize --eps 8 --seed 2018

python train_svm.py --dataset cifar10 --n_classes 2 --c 0.01 --target cifar10_svm_sinf.pkl --save  --binarize --eps 8 --seed 2018
python train_mlp.py --dataset cifar10 --n_classes 2 --hidden-nodes 20 --lr 0.01 --iters 200 --target cifar10_mlp_sinf.pkl --save  --binarize --eps 8 --seed 2018


python train_svm.py --dataset gtsrb_binary --n_classes 2 --c 0.01 --target gtsrb_binary_svm.pkl --save --seed 2018

python train_svm.py --dataset gtsrb_binary --n_classes 2 --c 0.01 --target gtsrb_binary_svm_s128.pkl --save  --binarize --eps 128 --seed 2018
python train_svm.py --dataset gtsrb_binary --n_classes 2 --c 0.01 --target gtsrb_binary_svm_s8.pkl --save  --binarize --eps 8 --seed 2018

python train_mlp.py --dataset gtsrb_binary --n_classes 2 --hidden-nodes 20 --lr 0.001 --iters 1000 --target gtsrb_binary_mlp.pkl --save --seed 2018
python train_mlp.py --dataset gtsrb_binary --n_classes 2 --hidden-nodes 20 --lr 0.001 --iters 1000 --target gtsrb_binary_mlp_s128.pkl --save  --binarize --eps 128 --seed 2018
python train_mlp.py --dataset gtsrb_binary --n_classes 2 --hidden-nodes 20 --lr 0.001 --iters 1000 --target gtsrb_binary_mlp_s8.pkl --save  --binarize --eps 8 --seed 2018
