import os
import sys
import numpy as np

part_1 = [
    '#!/bin/bash -l\n',
    '#SBATCH -p datasci,datasci3,datasci4\n',
    '#SBATCH -x node429,node430\n',
    '#SBATCH --job-name=cifar10_01\n',
    '#SBATCH --gres=gpu:1\n',
    '#SBATCH --mem=16G\n',
    '#SBATCH --cpus-per-task=1\n',
    'cd ..\n',

]

part_2 = [
    'gpu=0\n',
    '\n',
]

files = []

seeds = np.arange(32)
acts = ['sign', 'sigmoid', 'relu']
nbs = [1, 2]
dms = [0, 1]

variables = []
for i in acts:
    for j in nbs:
        for k in dms:
            for m in seeds:
                if not os.path.exists(os.path.join('../checkpoints/pt', f'cifar10_toy2_{i}_i1_bce_nb{j}_nw1_dm{k}_s2_fp32_32_{m}.pt')):
                    variables.append((i, j, k, m))

for i, params in enumerate(variables):
    # for i, (seed, target_class) in enumerate(variables):
    # for i, (votes, index) in enumerate(variables):
    # for i, (version, scale) in enumerate(variables):

    part_3 = [

        'python train_cnn01.py --nrows 0.15 --localit 1 '
        '--updated_conv_features 32 --updated_fc_features 128 --updated_fc_nodes 1 '
        '--updated_conv_nodes 1 --width 100 --normalize 1 '
        f'--percentile 1 --fail_count 1 --loss bce --act {params[0]} '
        f'--fc_diversity 1 --init normal --no_bias {params[1]} --scale 2 '
        f'--w-inc1 0.17 --w-inc2 0.2 --version toy2 --seed {params[3]} --iters 5000 '
        f'--dataset cifar10 --n_classes 2 --cnn 1 --divmean {params[2]} '
        f'--target cifar10_toy2_{params[0]}_i1_bce_nb{params[1]}_nw1_dm{params[2]}_s2_fp32_32_{params[3]}.pkl '
        f'--updated_fc_ratio 1 --updated_conv_ratio 1\n'


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

