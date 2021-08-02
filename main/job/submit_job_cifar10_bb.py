import os
import sys
import numpy as np

part_1 = [
    '#!/bin/bash -l\n',
    '#SBATCH -p datasci,datasci3,datasci4\n',
    '#SBATCH --job-name=cifar10_job\n',
    '#SBATCH --gres=gpu:1\n',
    '#SBATCH --mem=8G\n',
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

variables = []

# b = np.arange(32)
a = [ 0.0625, 0.125, 0.25]
# a = [0.25]
# a = [1, 2, 3, 4]
# b = [1, 4, 8, 16, 32, 64, 96]
# a = [
#     # 'cifar10_simclr_mlp',
#     #  'cifar10_simclr_scd01mlp_32_br02_h20_nr075_ni1000_i1_0',
#      'cifar10_resnet18.pkl']

b = [2019, 2393, 92382, 232, 12, 58, 954, 758, 451, 2015687]

for i in a:
    for j in b:
        variables.append((i, j))

for i, (epsilon, seed) in enumerate(variables):
# for i, epsilon in enumerate(a):
    
    part_3 = [



        # 'for seed in 2019 2393 92382 232 12 58 954 758 451 2015687\n',
        # 'do\n',
#         f'python bb_attack.py --epsilon {epsilon} --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
# --train-size 200 --target cifar10_toy2_i3_bce_div_nb_16 --random-sign 1 --seed {seed} --dataset cifar10 \
# --oracle-size 1024 --n_classes 2\n',
#         f'python bb_attack.py --epsilon {epsilon} --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
# --train-size 200 --target cifar10_toy2_i3_bce_div_nb_s2_16 --random-sign 1 --seed {seed} --dataset cifar10 \
# --oracle-size 1024 --n_classes 2\n',
#         f'python bb_attack.py --epsilon {epsilon} --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
# --train-size 200 --target cifar10_toy2_s2_bp --random-sign 1 --seed {seed} --dataset cifar10 \
# --oracle-size 1024 --n_classes 2\n',
#         f'python bb_attack.py --epsilon {epsilon} --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
# --train-size 200 --target cifar10_resnet18 --random-sign 1 --seed {seed} --dataset cifar10 \
# --oracle-size 1024 --n_classes 2\n',
#         f'python bb_attack.py --epsilon {epsilon} --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
# --train-size 200 --target cifar10_lenet --random-sign 1 --seed {seed} --dataset cifar10 \
# --oracle-size 1024 --n_classes 2\n',
#         f'python bb_attack.py --epsilon {epsilon} --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
# --train-size 200 --target cifar10_toy_i3_bce_16 --random-sign 1 --seed {seed} --dataset cifar10 \
# --oracle-size 1024 --n_classes 2\n',
#         f'python bb_attack.py --epsilon {epsilon} --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
# --train-size 200 --target cifar10_toy_i3_01_16 --random-sign 1 --seed {seed} --dataset cifar10 \
# --oracle-size 1024 --n_classes 2\n',
#         f'python bb_attack.py --epsilon {epsilon} --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
# --train-size 200 --target cifar10_toy_i3_bce_100 --random-sign 1 --seed {seed} --dataset cifar10 \
# --oracle-size 1024 --n_classes 2\n',
        f'python bb_attack.py --epsilon {epsilon} --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target cifar10_binary_toy2np_sg_i1_bce_div_nb_s2_16 --random-sign 1 --seed {seed} --dataset cifar10_binary \
--oracle-size 1024 --n_classes 2\n',
        # 'done\n',

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

