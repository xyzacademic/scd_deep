import os
import sys
import numpy as np

part_1 = [
    '#!/bin/bash -l\n',
    '#SBATCH -p datasci,datasci4,datasci3\n',
    '#SBATCH --job-name=cifar10_job\n',
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

# variables = [
# # ('mlp1', 'relu'),
# # ('mlp2', 'relu'),
# # ('mlp3', 'relu'),
# # ('mlp4', 'relu'),
# ('mlp1', 'sign'),
# ('mlp2', 'sign'),
# ('mlp3', 'sign'),
# ('mlp4', 'sign'),
# # ('mlp1', 'tanh'),
# # ('mlp2', 'tanh'),
# # ('mlp3', 'tanh'),
# # ('mlp4', 'tanh'),
#
# ]

nrows = [0.15]
local_iter = [3]
w_inc1 = [0.1]
updated_conv_features = [3]
updated_fc_nodes = [1]
updated_conv_nodes = [1, ]
width = [100]
normalize = [0, 1]
percentile = [1]
fail_count = [1]
loss = ['bce']
structure = ['toy', 'lenet']
fc_diversity = [0, 1]
conv_diversity = [0]
diversity_train_stop_iters = [3000]
diversity = [0]
init = ['normal', 'uniform']

variables = []
for a in percentile:
    for b in local_iter:
        for c in updated_conv_features:
            for d in updated_fc_nodes:
                for e in updated_conv_nodes:
                    for f in width:
                        for g in normalize:
                            for h in nrows:
                                for i in fail_count:
                                    for j in loss:
                                        for k in w_inc1:
                                            for l in structure:
                                                for m in fc_diversity:
                                                    for n in conv_diversity:
                                                        for o in diversity_train_stop_iters:
                                                            for p in diversity:
                                                                for q in init:
                                                                    variables.append((a, b, c, d, e,
                                                                                  f, g, h, i, j, k, l,
                                                                                  m, n, o, p, q))

for i, params in enumerate(variables):
    # if not params[12] and not params[13] and not params[15]:
    #     print('continue')
    #     continue

    part_3 = [
        f'python grid_search2.py --nrows {params[7]} --localit {params[1]} ' 
        f'--updated_conv_features {params[2]} --updated_fc_nodes {params[3]} '
        f'--updated_conv_nodes {params[4]} --width {params[5]} --normalize {params[6]} '
        f'--percentile {params[0]} --fail_count {params[8]} --loss {params[9]} '
        f'--w-inc1 {params[10]} --version {params[11]} --fc_diversity {params[12]} '
        f'--conv_diversity {params[13]} --diversity_train_stop_iters {params[14]} '
        f'--diversity {params[15]} --init {params[16]} --iters 3000\n'

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

