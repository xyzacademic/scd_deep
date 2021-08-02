import os
import sys
import numpy as np

part_1 = [
    '#!/bin/bash -l\n',
    '#SBATCH -p datasci,datasci3,datasci4\n',
    '#SBATCH --job-name=cifar10_hsj\n',
    # '#SBATCH --gres=gpu:1\n',
    '#SBATCH --mem=16G\n',
    '#SBATCH --cpus-per-task=2\n',
    'cd ..\n',

]

part_2 = [
    'gpu=0\n',
    '\n',
]

files = []

index = [0, 288, 5, 9, 6, 800, 1991, 1000, 1500, 1800]
# index = [288, 9, 6]
#index = [0, 5, 800, 1991, 1000, 1500, 1800]
models = [
#    '--target cifar10_toy2_s2_bp_100.pkl --cnn 1',
#    '--target cifar10_toy2_i1_bce_nb_div_s2_100.pkl --cnn 1',
#    '--target cifar10_toy2_i1_01_nb_div_s2_100.pkl --cnn 1',
    '--target cifar10_toy2_i1_bce_nb_div_s2_eps2_100.pkl --cnn 1',
    '--target cifar10_toy2_i1_01_nb_div_s2_eps2_100.pkl --cnn 1',


]

variables = []

for i in index:
    for j in models:
        variables.append((i, j))



# for i, (epsilon, seed) in enumerate(variables):
for i, (index, params) in enumerate(variables):

    part_3 = [
    'python hopskipjump_attack.py --dataset cifar10 --n_classes 2 --votes 16 --iters 100 '
    f'--train-size 10000 --index {index} --round 2 --adv_init 0 {params} --source hsj_rdn_attack_logs_100\n'

    # 'python hopskipjump_attack.py --dataset cifar10 --n_classes 2 --votes 16 --iters 100 '
    # f'--train-size 100 --index {index} --round 2 --adv_init 1 {params} --source hsj_fi_attack_logs_100\n'

            ]

    file_name = 'temp_sh/temp_job%d.sh' % i
    files.append(file_name)
    with open(file_name, 'w') as f:

        f.writelines(part_1)
        f.writelines(part_2)
        f.writelines(part_3)


with open('subjob.sh', 'w') as f:
    f.writelines(['#!/bin/bash\n\n'])
    for name in files:
        f.write('sbatch %s\n' % name)

