import os
import sys
import numpy as np

part_1 = [
    '#!/bin/bash -l\n',
    '#SBATCH -p datasci,datasci3,datasci4\n',
    '#SBATCH --job-name=cifar10_db\n',
    '#SBATCH -x node429,node430,node412\n',
    '#SBATCH --gres=gpu:1\n',
    '#SBATCH --mem=32G\n',
    '#SBATCH --cpus-per-task=4\n',
    'cd ..\n',

]

part_2 = [
    'gpu=0\n',
    '\n',
]

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



# for i, (epsilon, seed) in enumerate(variables):
# for i, (index, params) in enumerate(variables):
for i, params in enumerate(variables):
    indices = np.load(f'../intersection_index/{params[1]}_mlp_{params[0][0]}{params[0][1]}.npy')
    part_3 = [
        f'python decision_boundary_attack.py --dataset {params[1]} --n_classes 2 --votes 8 --iters 100 --cnn 0 '
        f'--train-size 100 --index {index} --round 1 --adv_init 1 --version mlp --c0 {params[0][0]} --c1 {params[0][1]} '
        f'--target {params[1]}_{params[0][0]}{params[0][1]}_{"mlpbcebp"}_8.pkl --source db_fi_attack_logs_100\n '

        f'python decision_boundary_attack.py --dataset {params[1]} --n_classes 2 --votes 8 --iters 100 --cnn 0 '
        f'--train-size 100 --index {index} --round 1 --adv_init 1 --version mlp --c0 {params[0][0]} --c1 {params[0][1]} '
        f'--target {params[1]}_{params[0][0]}{params[0][1]}_{"mlpbceban"}_8.pkl --source db_fi_attack_logs_100\n '

        f'python decision_boundary_attack.py --dataset {params[1]} --n_classes 2 --votes 8 --iters 100 --cnn 0 '
        f'--train-size 100 --index {index} --round 1 --adv_init 1 --version mlp --c0 {params[0][0]} --c1 {params[0][1]} '
        f'--target {params[1]}_{params[0][0]}{params[0][1]}_{"mlp01scd"}_8.pkl --source db_fi_attack_logs_100\n '

        f'python decision_boundary_attack.py --dataset {params[1]} --n_classes 2 --votes 8 --iters 100 --cnn 0 '
        f'--train-size 100 --index {index} --round 1 --adv_init 1 --version mlp --c0 {params[0][0]} --c1 {params[0][1]} '
        f'--target {params[1]}_{params[0][0]}{params[0][1]}_{"mlpbcescd"}_8.pkl --source db_fi_attack_logs_100\n '

     for index in indices

            ]

    # indices = np.load(f'../intersection_index/{params[1]}_cnn_{params[0][0]}{params[0][1]}.npy')
    # part_3 = [
    #     f'python decision_boundary_attack.py --dataset {params[1]} --n_classes 2 --votes 8 --iters 40 --cnn 1 '
    #     f'--train-size 100 --index {index} --round 1 --adv_init 1 --version cnn --c0 {params[0][0]} --c1 {params[0][1]} '
    #     f'--target {params[1]}_{params[0][0]}{params[0][1]}_{"toy3rrr100"}_8.pkl --source db_fi_attack_logs_100\n '
    #
    #     f'python decision_boundary_attack.py --dataset {params[1]} --n_classes 2 --votes 8 --iters 40 --cnn 1 '
    #     f'--train-size 100 --index {index} --round 1 --adv_init 1 --version cnn --c0 {params[0][0]} --c1 {params[0][1]} '
    #     f'--target {params[1]}_{params[0][0]}{params[0][1]}_{"toy3sss100scale_ban"}_8.pkl --source db_fi_attack_logs_100\n '
    #
    #     f'python decision_boundary_attack.py --dataset {params[1]} --n_classes 2 --votes 8 --iters 40 --cnn 1 '
    #     f'--train-size 100 --index {index} --round 1 --adv_init 1 --version cnn --c0 {params[0][0]} --c1 {params[0][1]} '
    #     f'--target {params[1]}_{params[0][0]}{params[0][1]}_{"toy3sss100scale_abp"}_8.pkl --source db_fi_attack_logs_100\n '
    #
    #
    #
    #     for index in indices
    #
    # ]

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

