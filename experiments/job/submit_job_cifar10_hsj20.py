import os
import sys
import numpy as np

part_1 = [
    '#!/bin/bash -l\n',
    '#SBATCH -p datasci,datasci3,datasci4\n',
    '#SBATCH --job-name=cifar10_hsj\n',
    '#SBATCH -x node429,node430\n',
    # '#SBATCH --gres=gpu:1\n',
    '#SBATCH --mem=16G\n',
    '#SBATCH --cpus-per-task=2\n',
    'cd ..\n',

]

part_2 = [
    'gpu=0\n',
    '\n',
]

files = []

index = [1760, 1716, 3480, 3628, 2961, 1117, 7466, 2791, 2502, 5332,
         1651, 351, 9234, 5033, 9074, 319, 777, 8074, 3950, 3979]
# index = np.load('../cifar10_interection_1000.npy')[100:]
# index = [288, 9, 6]
#index = [0, 5, 800, 1991, 1000, 1500, 1800]
models = [


# '--target cifar10_toy3rrr100_sign_i1_mce_b5000_lrc0.075_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_8.pkl --cnn 1',
# '--target cifar10_toy3rsr100_sign_i1_mce_b5000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_8.pkl --cnn 1',
# '--target cifar10_toy3rss100_sign_i1_mce_b5000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_8.pkl --cnn 1',
# '--target cifar10_toy3rrs100_bp_sign_i1_mce_b500_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_8.pkl --cnn 1',
# '--target cifar10_toy3rss100_bp_eps2_sign_i1_mce_b1000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl --cnn 1',
# '--target cifar10_toy3rsr100_bp_eps2_sign_i1_mce_b1000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl --cnn 1',
# '--target cifar10_toy3rrs100_bp_adv_sign_i1_mce_b1000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl --cnn 1',
# '--target toy3rrr_bp_eps2_fp16_8.pkl --cnn 1',
# '--target toy3rrr_bp_eps2_fp16_32.pkl --cnn 1',
# '--target cifar10_toy3rsrs100_bp_eps2_sign_i1_mce_b2500_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl --cnn 1',
# '--target cifar10_toy3rrss100_bp_adv_sign_i1_mce_b2500_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl --cnn 1',
# '--target cifar10_toy3rsss100_bp_eps2_sign_i1_mce_b2500_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl --cnn 1',
# '--target cifar10_toy3rrs100_bp_sign_i1_mce_b1000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl --cnn 1',
#     '--target cifar10_toy3rrr100_bp_adv_8.pkl --cnn 1',
#     '--target cifar10_toy3rrr100_bp_adv_32.pkl --cnn 1',
#     '--target toy3rrr_bp_fp16_eps2_8.pkl --cnn 1',
# '--target cifar10_toy3rrss100_adaptivebs_bp_adv_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl --cnn 1',
# '--target cifar10_toy3rrss100_adaptivebs_bp_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl --cnn 1',
# '--target cifar10_toy3rsss100_adaptivebs_bp_adv_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl --cnn 1',
#     '--target toy3rrr_bp_fp16_32.pkl --cnn 1',
#     '--target cifar10_toy3rsss100_adaptivebs_bp_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl --cnn 1',
#     '--target cifar10_toy3rrss100_adaptivebs_bp_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl --cnn 1',
#     '--target cifar10_toy3ssss100_adaptivebs_bp_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl --cnn 1',
#
#     '--target cifar10_toy3rrr100_bp_adv_32.pkl --cnn 1',
#     '--target cifar10_toy3rrss100_adaptivebs_bp_adv_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl --cnn 1',
#     '--target cifar10_toy3rsss100_adaptivebs_bp_adv_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl --cnn 1',
#     '--target cifar10_toy3ssss100_adaptivebs_bp_adv_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl --cnn 1',
#     '--target cifar10_toy3rrss100_adaptivebs_bp_sign_i1_01loss_b1000_lrc0.05_lrf0.01_nb2_nw0_dm0_upc1_upf10_ucf32_fp16_32.pkl --cnn 1',
    '--target cifar10_toy3rsss100_bp_sign_i1_01loss_b5000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf10_ucf32_fp16_32.pkl --cnn 1',
    '--target cifar10_toy3ssss100_bp_sign_i1_01loss_b5000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf10_ucf32_fp16_32.pkl --cnn 1',
    '--target cifar10_toy3rrss100_bp_sign_i1_01loss_b5000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf10_ucf32_fp16_32.pkl --cnn 1',

]

variables = []
#
for i in index:
    for j in models:
        variables.append((i, j))



# for i, (epsilon, seed) in enumerate(variables):
# for i, (index, params) in enumerate(variables):
for i, params in enumerate(models):
    part_3 = [
    # 'python hopskipjump_attack.py --dataset cifar10 --n_classes 10 --votes 32 --iters 100 '
    # f'--train-size 100 --index {index} --round 2 --adv_init 0 {params} --source hsj_rdn_attack_logs_mc_example\n'

    'python hopskipjump_attack_20.py --dataset cifar10 --n_classes 10 --votes 32 --iters 100 '
    f'--train-size 100 --index {1} --round 1 --adv_init 1 {params} --source hsj_fi_attack_logs_mc_example\n'

    # 'python hopskipjump_attack.py --dataset cifar10 --n_classes 10 --votes 32 --iters 40 '
    # f'--train-size 100 --index {index} --round 2 --adv_init 1 {model} --source hsj_fi_attack_logs_mc_1000\n' for model in models

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

