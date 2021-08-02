#!/bin/sh

#$-q datasci
#$-q datasci3
#$-cwd
#$-N eva

cd ..

python evaluation.py --dataset mnist --Lambda 0.1 --epsilon 0.3 --random-sign 1
python evaluation.py --dataset cifar10 --Lambda 0.1 --epsilon 0.0625 --random-sign 1