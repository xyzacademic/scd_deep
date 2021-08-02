#!/bin/sh

#$-q datasci
#$-q datasci3
#$-cwd
#$-N svm_attack

cd ..

python mnist_original_svm_01.py --epsilon 0.3 --Lambda 0.1 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.001 --train-size 200 --target mnist_svm --random-sign 1
#python mnist_original_svm_01.py --epsilon 0.3 --Lambda 0.1 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.001 --train-size 200 --target mnist_svm_adv --random-sign 1
#python mnist_original_svm_01.py --epsilon 0.3 --Lambda 0.1 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.001 --train-size 200 --target mnist_svm_boot --random-sign 1
python mnist_original_svm_01.py --epsilon 0.3 --Lambda 0.1 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.001 --train-size 200 --target mnist_mlp --random-sign 1

#python mnist_original_svm_01.py --epsilon 0.2 --Lambda 0.1 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.001 --train-size 200 --target mnist_svm --random-sign 1
#python mnist_original_svm_01.py --epsilon 0.2 --Lambda 0.1 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.001 --train-size 200 --target mnist_svm_adv --random-sign 1
#python mnist_original_svm_01.py --epsilon 0.2 --Lambda 0.1 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.001 --train-size 200 --target mnist_svm_boot --random-sign 1
#python mnist_original_svm_01.py --epsilon 0.2 --Lambda 0.1 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.001 --train-size 200 --target mnist_mlp --random-sign 1

#python mnist_original_svm_01.py --epsilon 0.1 --Lambda 0.1 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.001 --train-size 200 --target mnist_svm --random-sign 1
#python mnist_original_svm_01.py --epsilon 0.1 --Lambda 0.1 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.001 --train-size 200 --target mnist_svm_adv --random-sign 1
#python mnist_original_svm_01.py --epsilon 0.1 --Lambda 0.1 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.001 --train-size 200 --target mnist_svm_boot --random-sign 1
#python mnist_original_svm_01.py --epsilon 0.1 --Lambda 0.1 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.001 --train-size 200 --target mnist_mlp --random-sign 1