import os

title = ['#!/bin/sh\n\n\n',
         'cd ..\n\n\n']

class_pairs = []
for c0 in range(9):
    for c1 in range(c0+1, 10):
        class_pairs.append((c0, c1))


contents = []


# mlpbcebp
datasets = [('cifar10_binary', 'mlprr'), ('stl10_binary', 'mlpcr20scale')]

variables = []
for j in class_pairs:
    for k in datasets:
        variables.append((j, k))

for params in variables:
    line = f'python combine_vote_mlp.py --dataset {params[1][0]} --n_classes 2 ' \
       f'--c0 {params[0][0]} --c1 {params[0][1]} --save ' \
           f'--target {params[1][0]}_{params[0][0]}{params[0][1]}_mlpbcebp --votes 8 ' \
       f'--no_bias 0 --scale 1 --cnn 0 --act sign --version {params[1][1]}'
    contents.append(line + '\n\n')

# ban
datasets = [('cifar10_binary', 'mlp01scale'), ('stl10_binary', 'mlpcs20scale')]
variables = []
for j in class_pairs:
    for k in datasets:
        variables.append((j, k))

for params in variables:
    line = f'python combine_vote_mlp.py --dataset {params[1][0]} --n_classes 2 ' \
       f'--c0 {params[0][0]} --c1 {params[0][1]} --save ' \
           f'--target {params[1][0]}_{params[0][0]}{params[0][1]}_mlpbceban --votes 8 ' \
       f'--no_bias 0 --scale 1 --cnn 0 --act sign --version {params[1][1]}'
    contents.append(line + '\n\n')

# mlp01scd
datasets = [('cifar10_binary', 'mlp01scale'), ('stl10_binary', 'mlpcs20scale')]
variables = []
for j in class_pairs:
    for k in datasets:
        variables.append((j, k))

for params in variables:
    line = f'python combine_vote_mlp.py --dataset {params[1][0]} --n_classes 2 ' \
           f'--c0 {params[0][0]} --c1 {params[0][1]} --save ' \
           f'--target {params[1][0]}_{params[0][0]}{params[0][1]}_mlp01scd --votes 8 ' \
           f'--no_bias 0 --scale 1 --cnn 0 --act sign --version {params[1][1]}'
    contents.append(line + '\n\n')

# mlpbcescd
datasets = [('cifar10_binary', 'mlprr'), ('stl10_binary', 'mlpcr20scale')]

variables = []
for j in class_pairs:
    for k in datasets:
        variables.append((j, k))

for params in variables:
    line = f'python combine_vote_mlp.py --dataset {params[1][0]} --n_classes 2 ' \
       f'--c0 {params[0][0]} --c1 {params[0][1]} --save ' \
           f'--target {params[1][0]}_{params[0][0]}{params[0][1]}_mlpbcescd --votes 8 ' \
       f'--no_bias 0 --scale 1 --cnn 0 --act sign --version {params[1][1]}'
    contents.append(line + '\n\n')

with open('combine_vote.sh', 'w') as f:
    f.writelines(title)
    f.writelines(contents)
