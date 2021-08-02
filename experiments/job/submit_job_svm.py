import os
import sys
import numpy as np

part_1 = [
    '#!/bin/bash -l\n',
    '#SBATCH -p datasci3,datasci4,datasci\n',
    # '#SBATCH -x node429,node430,node412\n',
    '#SBATCH --job-name=svm\n',
    # '#SBATCH --gres=gpu:1\n',
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

version = ['toy3rrr100']

seeds = np.arange(1)
# nrows = [256, 512, 768, 1024]
nrows = [200]
# acts = ['sign', 'sigmoid', 'relu']
acts = ['sign']
nbs = [2]
dms = [0]
upc = [1]
upf = [1]
ucf = [32]
norms = [0]
# lr_convs = [0.1]
# lr_fcs = [0.2]
lr_convs = [0.1]
lr_fcs = [0.17]
inits = ['normal']
losses = ['01loss']

class_pairs = []
for c0 in range(9):
    for c1 in range(c0+1, 10):
        class_pairs.append((c0, c1))
datasets = ['mnist_binary', 'cifar10_binary', 'stl10_binary']
variables = []

for i in seeds:
    for j in class_pairs:
        for k in datasets:
            variables.append((i, j, k))

for i, params in enumerate(variables):
    # for i, (seed, target_class) in enumerate(variables):
    # for i, (votes, index) in enumerate(variables):
    # for i, (version, scale) in enumerate(variables):

    # # MNIST
    # part_3 = [
    #
    #
    # ]

    # CIFAR10
    part_3 = [
        f'python train_svm.py --dataset {params[2]} --c0 {params[1][0]} --c1 {params[1][1]} --c 0.01 '
        f'--target {params[2]}_{params[1][0]}{params[1][1]}_svm.pkl --save'

    ]
    #
    # # STL10
    # part_3 = [
    #
    #
    # ]
    #
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

