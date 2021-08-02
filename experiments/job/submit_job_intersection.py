import os
import sys
import numpy as np

part_1 = [
    '#!/bin/bash -l\n',
    '#SBATCH -p datasci3,datasci4,datasci\n',
    # '#SBATCH -x node429,node430,node412\n',
    '#SBATCH --job-name=ban\n',
    '#SBATCH --gres=gpu:1\n',
    '#SBATCH --mem=16G\n',
    '#SBATCH --cpus-per-task=2\n',
    'cd ..\n',

]

part_2 = [
    'gpu=0\n',
    'cpus=2\n',
    '\n',
]

if not os.path.exists('temp_sh'):
    os.makedirs('temp_sh')
files = []

class_pairs = []
for c0 in range(9):
    for c1 in range(c0+1, 10):
        class_pairs.append((c0, c1))
datasets = [
    'cifar10_binary',
    # 'stl10_binary'
]
variables = []


for j in class_pairs:
    for k in datasets:
        variables.append((j, k))

for i, params in enumerate(variables):

    part_3 = [
        #

        f'python get_intersection.py --dataset {params[1]} --n_classes 2 '
        f'--c0 {params[0][0]} --c1 {params[0][1]} --version mlp --cnn 0 \n'

        f'python get_intersection.py --dataset {params[1]} --n_classes 2 '
        f'--c0 {params[0][0]} --c1 {params[0][1]} --version cnn --cnn 1 \n'
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

