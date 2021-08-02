import os
import sys
import numpy as np

part_1 = [
    '#!/bin/bash -l\n',
    '#SBATCH -p datasci3,datasci4,datasci\n',
    # '#SBATCH -x node429,node430,node412\n',
    '#SBATCH --job-name=cnn01scd\n',
    '#SBATCH --gres=gpu:1\n',
    '#SBATCH --mem=16G\n',
    '#SBATCH --cpus-per-task=4\n',
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

seeds = np.arange(8)
# nrows = [256, 512, 768, 1024]
nrows = [200]
# acts = ['sign', 'sigmoid', 'relu']
acts = ['sign']
nbs = [2]
dms = [0]
upc = [1]
upf = [1]
ucf = [32]
norms = [1]
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
datasets = [
    # ('cifar10_binary', 'mlp01scale'),
    ('stl10_binary', 'mlp01scale'),
]
variables = []

for i in seeds:
    for j in class_pairs:
        for k in datasets:
            variables.append((i, j, k))

for i, params in enumerate(variables):
    # for i, (seed, target_class) in enumerate(variables):
    # for i, (votes, index) in enumerate(variables):
    # for i, (version, scale) in enumerate(variables):
    # if os.path.exists(f'../checkpoints/pt/{params[2][0]}_{params[1][0]}{params[1][1]}_{"mlp01scd"}_{params[0]}.pt'):
    #     print(f'{params[2][0]}_{params[1][0]}{params[1][1]}_{"mlp01scd"}_{params[0]}.pt exist')
    #     continue
    part_3 = [
        #
        # f'python train_bce.py --aug 1 --n_classes 2 --no_bias 1 --seed {params[0]} --version {"toy3rrr100"} --epoch 100 '
        # f'--target {params[2][0]}_{params[1][0]}{params[1][1]}_{"toy3rrr100"}_{params[0]} --save --batch-size 256 '
        # f'--dataset {params[2][0]} --c0 {params[1][0]} --c1 {params[1][1]} --lr 0.001 --n-jobs 4 \n'
        #
        # f'python train_bce.py --aug 1 --n_classes 2 --no_bias 1 --seed {params[0]} --version {"toy3sss100scale"} --epoch 100 '
        # f'--target {params[2][0]}_{params[1][0]}{params[1][1]}_{"toy3sss100scale"}_ban_{params[0]} --save --batch-size 256 '
        # f'--dataset {params[2][0]} --c0 {params[1][0]} --c1 {params[1][1]} --lr 0.001 --n-jobs 4 \n'

        f'python train_bce.py --aug 1 --n_classes 2 --no_bias 1 --seed {params[0]} --version {"toy4rrr100"} --epoch 200 '
        f'--target {params[2][0]}_{params[1][0]}{params[1][1]}_{"toy4rrr100"}_{params[0]} --save --batch-size 128 '
        f'--dataset {params[2][0]} --c0 {params[1][0]} --c1 {params[1][1]} --lr 0.001 --n-jobs 4 \n'

        f'python train_bce.py --aug 1 --n_classes 2 --no_bias 1 --seed {params[0]} --version {"toy4ssss100scale"} --epoch 200 '
        f'--target {params[2][0]}_{params[1][0]}{params[1][1]}_{"toy4ssss100scale"}_ban_{params[0]} --save --batch-size 128 '
        f'--dataset {params[2][0]} --c0 {params[1][0]} --c1 {params[1][1]} --lr 0.001 --n-jobs 4 \n'
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

