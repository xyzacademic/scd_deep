#!/bin/sh

#$-q datasci
#$-q datasci3
#$-q datasci4
#$-cwd
#$-N scd_attack

cd ..

python train_svm.py --dataset mnist --n_classes 10 --c 0.01 --target mnist_svm.pkl --save --seed 2018
python train_mlp.py --dataset mnist --n_classes 10 --hidden-nodes 20 --lr 0.01 --iters 200 --target mnist_mlp.pkl --save --seed 2018