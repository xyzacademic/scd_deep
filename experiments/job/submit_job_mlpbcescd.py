import os
import sys
import numpy as np

part_1 = [
    '#!/bin/bash -l\n',
    '#SBATCH -p datasci3,datasci4\n',
    # '#SBATCH -x node429,node430,node412\n',
    '#SBATCH --job-name=mlpbcescd\n',
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
datasets = [('cifar10_binary', 'mlprr'), ('stl10_binary', 'mlpcr20scale')]
variables = []

for i in seeds:
    for j in class_pairs:
        for k in datasets:
            variables.append((i, j, k))

for i, params in enumerate(variables):
    # for i, (seed, target_class) in enumerate(variables):
    # for i, (votes, index) in enumerate(variables):
    # for i, (version, scale) in enumerate(variables):

    part_3 = [
        #

        f'python train_cnn01_01.py --nrows {0.75} --localit 1 '
        f'--updated_fc_features 128 --updated_fc_nodes {1} '
        f'--width 100 --normalize 0 '
        f'--percentile 1 --fail_count 1 --loss bce --act sign '
        f'--fc_diversity 1 --init normal --no_bias 0 --scale 1 '
        f'--w-inc1 {0.17} --w-inc2 {0.17} --version {params[2][1]} --seed {params[0]} --iters 1000 '
        f'--dataset {params[2][0]} --n_classes 2 --cnn 0 --divmean 0 '
        f'--target {params[2][0]}_{params[1][0]}{params[1][1]}_{"mlpbcescd"}_{params[0]} '
        f'--updated_fc_ratio 1 --verbose_iter 50 --c0 {params[1][0]} --c1 {params[1][1]} \n '
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

