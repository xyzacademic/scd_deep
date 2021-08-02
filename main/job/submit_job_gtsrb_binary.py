import os
import sys
import numpy as np



part_1 = [
    '#!/bin/bash -l\n',
    '#SBATCH -p datasci\n',
    '#SBATCH --job-name=gtsrb_binary_job\n',
    '#SBATCH --gres=gpu:1\n',
    '#SBATCH --mem=16G\n',
    '#SBATCH --cpus-per-task=1\n',

    'cd ../\n',

]

part_2 = [
    'gpu=0\n',
    '\n',
]

files = []

variables = []
a = [
    # 'cifar10_scd01mlp_4_br02_nr025_ni1000_i1',
    'gtsrb_binary_32_nr025_mlp1_sign_01loss_1000_w1_h1',
    'gtsrb_binary_32_nr025_mlp2_sign_01loss_1000_w1_h1',
    'gtsrb_binary_32_nr025_mlp3_sign_01loss_1000_w1_h1',
    'gtsrb_binary_32_nr025_mlp4_sign_01loss_1000_w1_h1',
    # 
    # 'cifar10_binary_lenet_100',
    # 'cifar10_binary_lenet_100_ep1',
    # 'cifar10_binary_lenet_100_ep2',
    #
    #
    # 'cifar10_binary_simplenet_100',
    # 'cifar10_binary_simplenet_100_ep1',
    # 'cifar10_binary_simplenet_100_ep2',

    # 'cifar10_binary_scd01mlp_100_br02_h500_nr025_ni25000_i1',
    # 'cifar10_binary_scd01mlp_100_br02_h500_nr025_ni25000_i1_ep1.pkl',
    # 'cifar10_binary_scd01mlp_100_br02_h500_nr025_ni25000_i1_ep2.pkl',

    # 'gtsrb_binary_mlp_100',
    # 'gtsrb_binary_mlp_100_ep004',
    # 'gtsrb_binary_mlp_100_ep1',
    # 'gtsrb_binary_mlp_100_ep2',

    # 'cifar10_mlpbnn_approx',
    # 'cifar10_mlpbnn_approx_ep004',
    # 'cifar10_mlpbnn_approx_ep1',
    # 'cifar10_mlpbnn_approx_ep2',
    # 'mlp1',
    # 'mlp2',
    # 'mlp3',
    # 'mlp4',
]
# b = np.arange(32)
b = [0.015625, 0.03125, 0.0625, 0.125, 0.25]
for i in a:
    for j in b:
        variables.append((i, j))

# for i, (version, epsilon) in enumerate(variables):
for i in range(100):
    seed = i
    # variable_key = 2 * (i+1)
    # variable_1 = variable_key
    # variable_2 = variable_key / 10
    # seed=2018

    part_3 = [

        'python train_mlp_logistic_ml_2.py --nrows 0.25 --nfeatures 1 --w-inc1 0.05 --w-inc2 0.05 --hidden-nodes 20 --num-iters 1000 \
        --b-ratio 0.2 --updated-features 256 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 1 \
        --target gtsrb_binary_1_nr025_mlp1_sign_logistic_1000_%d_w1_h1_005005.pkl --dataset gtsrb_binary  --seed %d  \
        --version mlp1 --act sign --width 1 --status sigmoid\n' % (seed, seed),

        'python train_mlplogistic.py --nrows 0.25 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
        --b-ratio 0.2 --updated-features 256 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 1 \
        --target gtsrb_binary_1_nr025_mlp1_sign_logistic_1000_%d_w1_h1.pkl --dataset gtsrb_binary  --seed %d --act sign \
         --width 1 --status sigmoid\n' % (seed, seed)

#         'for seed in 2019 2393 92382 232 12 58 954 258 451 2015687\n',
#         'do\n',
#         'python bb_attack.py --epsilon %f --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
# --train-size 200 --target %s --random-sign 1 --seed $seed --dataset gtsrb_binary \
# --oracle-size 1024 --n_classes 2\n' % (epsilon, version),
#         'done\n',

#         'for seed in 2019 2393 92382 232 12 58 954 258 451 2015687\n',
#         'do\n',
#         'python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
# --train-size 200 --target cifar10_1_nr025_%s_%s_01loss_1000_2018_w1_h1_005005 --random-sign 1 --seed $seed --dataset cifar10 \
# --oracle-size 1024 --n_classes 2\n' % (version, act),
#         'done\n',
#         'python train_mlp_ml_.py --nrows 0.25 --nfeatures 1 --w-inc1 0.1 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
# --b-ratio 0.2 --updated-features 256 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 1 \
# --target gtsrb_binary_1_nr025_%s_sign_01loss_1000_%d_w1_h1.pkl --dataset gtsrb_binary  --seed %d  --version %s \
# --act sign --width 1' % (version, seed, seed, version)

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

