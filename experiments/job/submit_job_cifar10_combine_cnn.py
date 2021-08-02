import os
import sys
import numpy as np

part_1 = [
    '#!/bin/bash -l\n',
    '#SBATCH -p datasci,datasci3,datasci4\n',
    '#SBATCH --job-name=cifar10_div\n',
    # '#SBATCH --gres=gpu:1\n',
    '#SBATCH --mem=24G\n',
    '#SBATCH --cpus-per-task=4\n',
    'cd ..\n',

]

part_2 = [
    'gpu=0\n',
    '\n',
]

files = []

votes = [8]
acts = ['sign']
nbs = [0]
dms = [0]
nws = [0]
losses = ['01loss']
nrows = [7500]
# version = ['toy3srr100scale', 'toy3ssr100scale', 'toy3sss100scale', 'toy3ssss100scale']
# version = ['toy4srr100scale', 'toy4ssr100scale', 'toy4sss100scale', 'toy4ssss100scale']
# version = ['mlpcs20scale']
version = ['mlp01scale']
# nrows = [1000]
# version = [ 'toy3rrs100']
upc = [1]
upf = [1]
ucf = [32]
lr_convs = [0.05, 0.025, 0.1, 0.075, 0.17]
lr_fcs = [0.7, 0.025, 0.05, 0.1, 0.17]


variables = []
for i in acts:
    for j in nbs:
        for k in dms:
            for o in upc:
                for p in upf:
                    for q in version:
                        for r in nrows:
                            for s in ucf:
                                for lc in lr_convs:
                                    for lf in lr_fcs:
                                        for m in votes:
                                            for loss in losses:
                                                for norm in nws:
                                                    target = f'cifar10_binary_{q}_abp_{i}_i1_{loss}_b{r}_lrc{lc}_lrf{lf}_nb{j}_nw{norm}_dm{k}_upc{o}_upf{p}_ucf{s}_normal'
                                                    # target = f'cifar10_binary_{q}_nb2_{loss}_bp_fp32'
                                                    if os.path.exists(os.path.join('../checkpoints/pt', f'{target}_{votes[-1]-1}.pt')):

                                                        variables.append((i, m, q, target))


for i, params in enumerate(variables):
    part_3 = [

        # f'python combine_vote_new.py --dataset cifar10 '
        # f'--n_classes 2 --votes {params[1]} --no_bias 1 '
        # f'--scale 1 --cnn 1 --version {params[2]} --act {params[0]} '
        # f'--target {params[3]} --save \n'

        # f'python combine_vote_mlp.py --dataset cifar10 '
        # f'--n_classes 2 --votes {params[1]} --no_bias 1 '
        # f'--scale 1 --cnn 1 --version {params[2]} --act {params[0]} '
        # f'--target {params[3]} --save \n'

        # f'python combine_vote_mlp.py --dataset cifar10 '
        # f'--n_classes 2 --votes {params[1]} --no_bias 0 '
        # f'--scale 1 --cnn 0 --version {params[2]} --act {params[0]} '
        # f'--target {params[3]} --save --name {params[3]} --seed {i}\n'  for i in range(8)

        f'python combine_vote_mlp.py --dataset cifar10 '
        f'--n_classes 2 --votes {params[1]} --no_bias 0 '
        f'--scale 1 --cnn 1 --version {params[2]} --act {params[0]} '
        f'--target {params[3]} --save \n'
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

