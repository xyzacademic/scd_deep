import os
import sys
import numpy as np

part_1 = [
    '#!/bin/bash -l\n',
    '#SBATCH -p datasci\n',
    '#SBATCH --job-name=imagenet_job\n',
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

variables = []
a = [
    'imagenet_32_nr025_mlp1_sign_01loss_1000_w1_h1',
    'imagenet_32_nr025_mlp2_sign_01loss_1000_w1_h1',
    'imagenet_32_nr025_mlp3_sign_01loss_1000_w1_h1',
    'imagenet_32_nr025_mlp4_sign_01loss_1000_w1_h1',
    # 
    # 'imagenet_binary_lenet_100',
    # 'imagenet_binary_lenet_100_ep1',
    # 'imagenet_binary_lenet_100_ep2',
    #
    #
    # 'imagenet_binary_simplenet_100',
    # 'imagenet_binary_simplenet_100_ep1',
    # 'imagenet_binary_simplenet_100_ep2',

    # 'imagenet_binary_scd01mlp_100_br02_h500_nr075_ni25000_i1',
    # 'imagenet_binary_scd01mlp_100_br02_h500_nr075_ni25000_i1_ep1.pkl',
    # 'imagenet_binary_scd01mlp_100_br02_h500_nr075_ni25000_i1_ep2.pkl',
    
    # 'gtsrb_binary_mlp_100',
    # 'gtsrb_binary_mlp_100_ep004',
    # 'gtsrb_binary_mlp_100_ep1',
    # 'gtsrb_binary_mlp_100_ep2',

    # 'imagenet_mlpbnn_approx',
    # 'imagenet_mlpbnn_approx_ep004',
    # 'imagenet_mlpbnn_approx_ep1',
    # 'imagenet_mlpbnn_approx_ep2',
    # 'mlp1',
# 'mlp2',
# 'mlp3',
# 'mlp4',
]
# b = np.arange(32)
# b = [0.015625, 0.03125, 0.0625, 0.125, 0.25]
a = [1, 2, 3, 4]
b = [1, 4, 8, 16, 32, 64, 96]
for i in a:
    for j in b:
        variables.append((i, j))

# for i, seed in enumerate(variables):
for i in range(100):
    seed = i
    # variable_key = 2 * (i+1)
    # variable_1 = variable_key
    # variable_2 = variable_key / 10
    # seed=2018
    # h = version
    # vote = seed
    part_3 = [


    # 'python train_scd4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 --b-ratio 0.5 \
    # --updated-features 256 --round 1 --interval 20 --n-jobs 1 --num-gpus 1  --save --n_classes 2 \
    # --target imagenet_scd01mlp_1_br05_nr075_ni1000_%d.pkl --dataset imagenet --version mlp --seed %d --width 1500 \
    # --metrics balanced --init normal' % (seed, seed)


        'python train_mlp_logistic_ml_2.py --nrows 0.75 --nfeatures 1 --w-inc1 0.05 --w-inc2 0.05 --hidden-nodes 20 --num-iters 1000 \
        --b-ratio 0.2 --updated-features 256 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 1 \
        --target imagenet_1_nr075_mlp1_sign_logistic_1000_%d_w1_h1_005005.pkl --dataset imagenet  --seed %d  \
        --version mlp1 --act sign --width 1 --status sigmoid\n' % (seed, seed),

        'python train_mlplogistic.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
        --b-ratio 0.2 --updated-features 256 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 1 \
        --target imagenet_1_nr075_mlp1_sign_logistic_1000_%d_w1_h1.pkl --dataset imagenet  --seed %d --act sign \
         --width 1 --status sigmoid\n' % (seed, seed)


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

