import os
import sys
import numpy as np

part_1 = [
    '#!/bin/bash -l\n',
    '#SBATCH -p datasci3\n',
    '#SBATCH -x node429,node430,node412\n',
    '#SBATCH --job-name=cifar10_job\n',
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
    # 'cifar10_binary',
    'stl10_binary'
]
variables = []

for i in ['fgsm', 'pgd']:
    for j in class_pairs:
        for k in datasets:
            variables.append((i, j, k))



for i, params in enumerate(variables):

    part_3 = [
        # f'python mlp_attack_out.py --epsilon 16 --num-steps 20 --attack_type {params[0]} '
        # f'--dataset {params[2]} --c0 {params[1][0]} --c1 {params[1][1]} '
        # f'--name results/outer/{params[2]}/mlp_{params[0]}_16_20_{params[1][0]}{params[1][1]}\n'
        #
        # f'python mlp_attack_in.py --epsilon 16 --num-steps 20 --attack_type {params[0]} --votes 8 '
        # f'--dataset {params[2]} --c0 {params[1][0]} --c1 {params[1][1]} --target mlprr --source mlpbcebp '
        # f'--name results/inner/{params[2]}/mlpbcebp_{params[0]}_16_20_{params[1][0]}{params[1][1]}\n'
        #
        # f'python mlp_attack_in.py --epsilon 16 --num-steps 20 --attack_type {params[0]} --votes 8 '
        # f'--dataset {params[2]} --c0 {params[1][0]} --c1 {params[1][1]} --target mlp01scale --source mlpbceban '
        # f'--name results/inner/{params[2]}/mlpbceban_{params[0]}_16_20_{params[1][0]}{params[1][1]}\n'
        #
        # f'python mlp_attack_in.py --epsilon 16 --num-steps 20 --attack_type {params[0]} --votes 8 '
        # f'--dataset {params[2]} --c0 {params[1][0]} --c1 {params[1][1]} --target mlp01scale --source mlp01scd '
        # f'--name results/inner/{params[2]}/mlp01scd_{params[0]}_16_20_{params[1][0]}{params[1][1]}\n'
        #
        # f'python mlp_attack_in.py --epsilon 16 --num-steps 20 --attack_type {params[0]} --votes 8 '
        # f'--dataset {params[2]} --c0 {params[1][0]} --c1 {params[1][1]} --target mlprr --source mlpbcescd '
        # f'--name results/inner/{params[2]}/mlpbcescd_{params[0]}_16_20_{params[1][0]}{params[1][1]}\n'

        # f'python mlp_attack_in.py --epsilon 16 --num-steps 20 --attack_type {params[0]} --votes 8 '
        # f'--dataset {params[2]} --c0 {params[1][0]} --c1 {params[1][1]} --target mlpcr20scale --source mlpbcebp '
        # f'--name results/inner/{params[2]}/mlpbcebp_{params[0]}_16_20_{params[1][0]}{params[1][1]}\n'
        #
        # f'python mlp_attack_in.py --epsilon 16 --num-steps 20 --attack_type {params[0]} --votes 8 '
        # f'--dataset {params[2]} --c0 {params[1][0]} --c1 {params[1][1]} --target mlpcs20scale --source mlpbceban '
        # f'--name results/inner/{params[2]}/mlpbceban_{params[0]}_16_20_{params[1][0]}{params[1][1]}\n'
        #
        # f'python mlp_attack_in.py --epsilon 16 --num-steps 20 --attack_type {params[0]} --votes 8 '
        # f'--dataset {params[2]} --c0 {params[1][0]} --c1 {params[1][1]} --target mlpcs20scale --source mlp01scd '
        # f'--name results/inner/{params[2]}/mlp01scd_{params[0]}_16_20_{params[1][0]}{params[1][1]}\n'
        #
        # f'python mlp_attack_in.py --epsilon 16 --num-steps 20 --attack_type {params[0]} --votes 8 '
        # f'--dataset {params[2]} --c0 {params[1][0]} --c1 {params[1][1]} --target mlpcr20scale --source mlpbcescd '
        # f'--name results/inner/{params[2]}/mlpbcescd_{params[0]}_16_20_{params[1][0]}{params[1][1]}\n'

        # f'python cnn_attack_out.py --epsilon 16 --num-steps 20 --attack_type {params[0]} '
        # f'--dataset {params[2]} --c0 {params[1][0]} --c1 {params[1][1]} --cnn 1 '
        # f'--name results/outer/{params[2]}/cnn_{params[0]}_16_20_{params[1][0]}{params[1][1]}\n'
        #
        # f'python cnn_attack_in.py --epsilon 16 --num-steps 20 --attack_type {params[0]} --votes 8 --cnn 1  '
        # f'--dataset {params[2]} --c0 {params[1][0]} --c1 {params[1][1]} --target toy3sss100scale --source toy3sss100scale_ban '
        # f'--name results/inner/{params[2]}/toy3sss100scale_ban_{params[0]}_16_20_{params[1][0]}{params[1][1]}\n'
        #
        # f'python cnn_attack_in.py --epsilon 16 --num-steps 20 --attack_type {params[0]} --votes 8 --cnn 1  '
        # f'--dataset {params[2]} --c0 {params[1][0]} --c1 {params[1][1]} --target toy3sss100scale --source toy3sss100scale_abp '
        # f'--name results/inner/{params[2]}/toy3sss100scale_abp_{params[0]}_16_20_{params[1][0]}{params[1][1]}\n'
        #
        # f'python cnn_attack_in.py --epsilon 16 --num-steps 20 --attack_type {params[0]} --votes 8 --cnn 1  '
        # f'--dataset {params[2]} --c0 {params[1][0]} --c1 {params[1][1]} --target toy3rrr100 --source toy3rrr100 '
        # f'--name results/inner/{params[2]}/toy3rrr100_{params[0]}_16_20_{params[1][0]}{params[1][1]}\n'

        f'python cnn_attack_out.py --epsilon 16 --num-steps 20 --attack_type {params[0]} '
        f'--dataset {params[2]} --c0 {params[1][0]} --c1 {params[1][1]} --cnn 1 '
        f'--name results/outer/{params[2]}/cnn_{params[0]}_16_20_{params[1][0]}{params[1][1]}\n'

        f'python cnn_attack_in.py --epsilon 16 --num-steps 20 --attack_type {params[0]} --votes 8 --cnn 1  '
        f'--dataset {params[2]} --c0 {params[1][0]} --c1 {params[1][1]} --target toy4ssss100scale --source toy4ssss100scale_ban '
        f'--name results/inner/{params[2]}/toy4ssss100scale_ban_{params[0]}_16_20_{params[1][0]}{params[1][1]}\n'

        f'python cnn_attack_in.py --epsilon 16 --num-steps 20 --attack_type {params[0]} --votes 8 --cnn 1  '
        f'--dataset {params[2]} --c0 {params[1][0]} --c1 {params[1][1]} --target toy4ssss100scale --source toy4ssss100scale_abp '
        f'--name results/inner/{params[2]}/toy4ssss100scale_abp_{params[0]}_16_20_{params[1][0]}{params[1][1]}\n'

        f'python cnn_attack_in.py --epsilon 16 --num-steps 20 --attack_type {params[0]} --votes 8 --cnn 1  '
        f'--dataset {params[2]} --c0 {params[1][0]} --c1 {params[1][1]} --target toy4rrr100 --source toy4rrr100 '
        f'--name results/inner/{params[2]}/toy4rrr100_{params[0]}_16_20_{params[1][0]}{params[1][1]}\n'
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

