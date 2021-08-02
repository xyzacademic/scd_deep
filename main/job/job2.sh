#!/bin/bash -l

#SBATCH -p datasci4


#SBATCH --job-name=cifar10_job
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

cd ..

#python evaluation_summary.py --dataset gtsrb_binary --target 1 --seed 2020

#python evaluation_summary.py --dataset gtsrb_binary --target 2 --seed 2020

#python evaluation_summary.py --dataset gtsrb_binary --target 3 --seed 2020

#python evaluation_summary.py --dataset gtsrb_binary --target 4 --seed 2020

#python evaluation_summary.py --dataset gtsrb_binary --target 5 --seed 2020
#
python evaluation_summary.py --dataset gtsrb_binary --target 6 --seed 2020
