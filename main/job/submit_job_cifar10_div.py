import os
import sys
import numpy as np

part_1 = [
    '#!/bin/bash -l\n',
    '#SBATCH -p datasci,datasci3,datasci4\n',
    '#SBATCH --job-name=cifar10_div\n',
    # '#SBATCH --gres=gpu:1\n',
    '#SBATCH --mem=16G\n',
    '#SBATCH --cpus-per-task=4\n',
    'cd ..\n',

]

part_2 = [
    'gpu=0\n',
    '\n',
]

files = []

seeds = np.arange(32)
acts = ['sign', 'sigmoid', 'relu']
nbs = [0, 1, 2]
dms = [0, 1]

variables = []
for i in acts:
    for j in nbs:
        for k in dms:
            variables.append((i, j, k))


for i, params in enumerate(variables):

    part_3 = [

        f'python combine_vote.py --dataset cifar10 --n_classes 2 --votes 32 '
        f'--scale 1 --cnn 0 --version fc --act {params[0]} '
        f'--target cifar10_fc_{params[0]}_i1_bce_nb{params[1]}_nw1_dm{params[2]}_s1_fp32_32\n'

        f'python combine_vote.py --dataset cifar10 --n_classes 2 --votes 32 '
        f'--scale 2 --cnn 1 --version toy2 --act {params[0]} '
        f'--target cifar10_toy2_{params[0]}_i1_bce_nb{params[1]}_nw1_dm{params[2]}_s2_fp32_32\n'


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

