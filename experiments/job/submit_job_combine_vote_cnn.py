import os

title = [
    '#!/bin/bash -l\n',
    '#SBATCH -p datasci3,datasci4,datasci\n',
    # '#SBATCH -x node429,node430,node412\n',
    '#SBATCH --job-name=combine_vote_cnn\n',
    # '#SBATCH --gres=gpu:1\n',
    '#SBATCH --mem=16G\n',
    '#SBATCH --cpus-per-task=8\n',

    'cd .. \n'
         ]

class_pairs = []
for c0 in range(9):
    for c1 in range(c0+1, 10):
        class_pairs.append((c0, c1))


contents = []


# # mlpbcebp
# datasets = [('cifar10_binary', 'toy3rrr100'), ('stl10_binary', 'toy4rrr100')]
#
# variables = []
# for j in class_pairs:
#     for k in datasets:
#         variables.append((j, k))
#
# for params in variables:
#     line = f'python combine_vote_mlp.py --dataset {params[1][0]} --n_classes 2 ' \
#        f'--c0 {params[0][0]} --c1 {params[0][1]} --save ' \
#            f'--target {params[1][0]}_{params[0][0]}{params[0][1]}_{params[1][1]} --votes 8 ' \
#        f'--no_bias 1 --scale 1 --cnn 1 --act sign --version {params[1][1]}'
#     contents.append(line + '\n\n')
#
# # ban
# datasets = [('cifar10_binary', 'toy3sss100scale'), ('stl10_binary', 'toy4ssss100scale')]
# variables = []
# for j in class_pairs:
#     for k in datasets:
#         variables.append((j, k))
#
# for params in variables:
#     line = f'python combine_vote_mlp.py --dataset {params[1][0]} --n_classes 2 ' \
#        f'--c0 {params[0][0]} --c1 {params[0][1]} --save ' \
#            f'--target {params[1][0]}_{params[0][0]}{params[0][1]}_{params[1][1]}_ban --votes 8 ' \
#        f'--no_bias 1 --scale 1 --cnn 1 --act sign --version {params[1][1]}'
#     contents.append(line + '\n\n')

# cnn01
datasets = [
    # ('cifar10_binary', 'toy3sss100scale'),
    ('stl10_binary', 'toy4ssss100scale')
]
variables = []
for j in class_pairs:
    for k in datasets:
        variables.append((j, k))

for params in variables:
    line = f'python combine_vote_mlp.py --dataset {params[1][0]} --n_classes 2 ' \
       f'--c0 {params[0][0]} --c1 {params[0][1]} --save ' \
           f'--target {params[1][0]}_{params[0][0]}{params[0][1]}_{params[1][1]}_abp --votes 8 ' \
       f'--no_bias 1 --scale 1 --cnn 1 --act sign --version {params[1][1]}'
    contents.append(line + '\n\n')

# # mlp01scd
# datasets = [('cifar10_binary', 'mlp01scale'), ('stl10_binary', 'mlpcs20scale')]
# variables = []
# for j in class_pairs:
#     for k in datasets:
#         variables.append((j, k))
#
# for params in variables:
#     line = f'python combine_vote_mlp.py --dataset {params[1][0]} --n_classes 2 ' \
#            f'--c0 {params[0][0]} --c1 {params[0][1]} --save ' \
#            f'--target {params[1][0]}_{params[0][0]}{params[0][1]}_mlp01scd --votes 8 ' \
#            f'--no_bias 0 --scale 1 --cnn 0 --act sign --version {params[1][1]}'
#     contents.append(line + '\n\n')
#
# # mlpbcescd
# datasets = [('cifar10_binary', 'mlprr'), ('stl10_binary', 'mlpcr20scale')]
#
# variables = []
# for j in class_pairs:
#     for k in datasets:
#         variables.append((j, k))
#
# for params in variables:
#     line = f'python combine_vote_mlp.py --dataset {params[1][0]} --n_classes 2 ' \
#        f'--c0 {params[0][0]} --c1 {params[0][1]} --save ' \
#            f'--target {params[1][0]}_{params[0][0]}{params[0][1]}_mlpbcescd --votes 8 ' \
#        f'--no_bias 0 --scale 1 --cnn 0 --act sign --version {params[1][1]}'
#     contents.append(line + '\n\n')

with open('combine_vote.sh', 'w') as f:
    f.writelines(title)
    f.writelines(contents)
