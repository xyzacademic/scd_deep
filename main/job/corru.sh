#!/bin/bash -l

#SBATCH -p datasci

#SBATCH --workdir='.'
#SBATCH --job-name=pwa
#SBATCH --mem=14G


cd ..




python evaluate_corruption.py --dataset mnist --n_classes 10

python evaluate_corruption.py --dataset cifar10 --n_classes 10
#python evaluate_corruption.py --dataset imagenet --n_classes 10
#
#
#python evaluate_fp.py --dataset cifar10 --n_classes 10
#python evaluate_fp.py --dataset imagenet --n_classes 10