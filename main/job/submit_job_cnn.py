import os
import sys
import numpy as np

part_1 = [
    '#!/bin/bash -l\n',
    '#SBATCH -p datasci,datasci4,datasci3\n',
    '#SBATCH --job-name=cifar10_job\n',
    '#SBATCH --gres=gpu:1\n',
    '#SBATCH --mem=16G\n',
    '#SBATCH --cpus-per-task=4\n',
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

# version = ['resnet18', 'resnet50']
model_source = ['resnet18', 'resnet50']
model_target = ['resnet18', 'resnet50']
aug_source = [0, 1]
aug_target = [0, 1]
normalize_souce = [0, 1]
normalize_target = [0, 1]


variables = []
# for a in version:
for a in model_source:
    for b in model_target:
        for c in aug_source:
            for d in aug_target:
                for e in normalize_souce:
                    for f in normalize_target:

                        source_model = f'cifar10_mc_{a}_aug{c}_normalize{e}.pt'
                        target_model = f'cifar10_mc_{b}_aug{d}_normalize{f}.pt'
                        if source_model != target_model:
                            variables.append((source_model, target_model, a))

for i, params in enumerate(variables):

    part_3 = [
        # f'python train_cifar10.py --version {params[0]} '
        # f'--target cifar10_mc_{params[0]}_aug{params[1]}_normalize{params[2]} '
        # f'--gpu 0 --aug {params[1]} --normalize {params[2]} --batch-size 256 \n'
        f'python sgm_attack.py --source {params[0]} --target {params[1]} --arch {params[2]} '
        f'--epsilon 8 \n'
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

