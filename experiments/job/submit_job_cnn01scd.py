import os
import sys
import numpy as np

part_1 = [
    '#!/bin/bash -l\n',
    '#SBATCH -p datasci3,datasci4\n',
    # '#SBATCH -x node429,node430,node412\n',
    '#SBATCH --job-name=cnn01scd\n',
    '#SBATCH --gres=gpu:1\n',
    '#SBATCH --mem=16G\n',
    '#SBATCH --cpus-per-task=4\n',
    'cd ..\n',

]

part_2 = [
    'gpu=0\n',
    'cpus=2\n',
    '\n',
]

if not os.path.exists('temp_sh'):
    os.makedirs('temp_sh')
files = []


version = ['toy3rrr100']

seeds = np.arange(8)
# nrows = [256, 512, 768, 1024]
nrows = [200]
# acts = ['sign', 'sigmoid', 'relu']
acts = ['sign']
nbs = [2]
dms = [0]
upc = [1]
upf = [1]
ucf = [32]
norms = [1]
# lr_convs = [0.1]
# lr_fcs = [0.2]
lr_convs = [0.1]
lr_fcs = [0.17]
inits = ['normal']
losses = ['01loss']
class_pairs = []
for c0 in range(9):
    for c1 in range(c0+1, 10):
        class_pairs.append((c0, c1))
datasets = [
    # ('cifar10_binary', 'mlp01scale'),
    ('stl10_binary', 'mlp01scale'),
]
variables = []

for i in seeds:
    for j in class_pairs:
        for k in datasets:
            variables.append((i, j, k))

for i, params in enumerate(variables):
    # for i, (seed, target_class) in enumerate(variables):
    # for i, (votes, index) in enumerate(variables):
    # for i, (version, scale) in enumerate(variables):
    # if os.path.exists(f'../checkpoints/pt/{params[2][0]}_{params[1][0]}{params[1][1]}_{"mlp01scd"}_{params[0]}.pt'):
    #     print(f'{params[2][0]}_{params[1][0]}{params[1][1]}_{"mlp01scd"}_{params[0]}.pt exist')
    #     continue
    part_3 = [
        #


        # f'python train_cnn01_01.py --nrows {200 / 10000} --localit 1 '
        # f'--updated_conv_features 32 --updated_fc_features 128 --updated_fc_nodes 1 '
        # f'--updated_conv_nodes 1 --width 100 --normalize 0 '
        # f'--percentile 1 --fail_count 1 --loss 01loss --act sign '
        # f'--fc_diversity 1 --init normal --no_bias 2 --scale 1 '
        # f'--w-inc1 {0.025} --w-inc2 {0.1} --version {"toy3srr100scale"} --seed {params[0]} --iters 15000 '
        # f'--dataset {params[2][0]} --n_classes 2 --cnn 1 --divmean 0 '
        # f'--target {params[2][0]}_{params[1][0]}{params[1][1]}_{"toy3srr100scale"}_abp_{params[0]} '
        # f'--updated_fc_ratio 10 --updated_conv_ratio 20 --verbose_iter 500 --freeze_layer 0 '
        # f'--lr 0.001 --bp_layer 4 --aug 1 --c0 {params[1][0]} --c1 {params[1][1]}\n'
        # #
        # f'python train_cnn01_01.py --nrows {200 / 10000} --localit 1 '
        # f'--updated_conv_features 32 --updated_fc_features 128 --updated_fc_nodes 1 '
        # f'--updated_conv_nodes 1 --width 100 --normalize 0 '
        # f'--percentile 1 --fail_count 1 --loss 01loss --act sign '
        # f'--fc_diversity 1 --init normal --no_bias 2 --scale 1 '
        # f'--w-inc1 {0.025} --w-inc2 {0.05} --version {"toy3ssr100scale"} --seed {params[0]} --iters 15000 '
        # f'--dataset {params[2][0]} --n_classes 2 --cnn 1 --divmean 0 '
        # f'--target {params[2][0]}_{params[1][0]}{params[1][1]}_{"toy3ssr100scale"}_abp_{params[0]} '
        # f'--updated_fc_ratio 10 --updated_conv_ratio 20 --verbose_iter 500 --freeze_layer 1 '
        # f'--resume --source {params[2][0]}_{params[1][0]}{params[1][1]}_{"toy3srr100scale"}_abp_{params[0]} '
        # f'--lr 0.001 --bp_layer 3  --aug 1  --c0 {params[1][0]} --c1 {params[1][1]}\n'
        #
        # f'python train_cnn01_01.py --nrows {200 / 10000} --localit 1 '
        # f'--updated_conv_features 32 --updated_fc_features 128 --updated_fc_nodes 1 '
        # f'--updated_conv_nodes 1 --width 100 --normalize 0 '
        # f'--percentile 1 --fail_count 1 --loss 01loss --act sign '
        # f'--fc_diversity 1 --init normal --no_bias 2 --scale 1 '
        # f'--w-inc1 {0.05} --w-inc2 {0.05} --version {"toy3sss100scale"} --seed {params[0]} --iters 15000 '
        # f'--dataset {params[2][0]} --n_classes 2 --cnn 1 --divmean 0 '
        # f'--target {params[2][0]}_{params[1][0]}{params[1][1]}_{"toy3sss100scale"}_abp_{params[0]} '
        # f'--updated_fc_ratio 10 --updated_conv_ratio 20 --verbose_iter 500 --freeze_layer 2 '
        # f'--resume --source {params[2][0]}_{params[1][0]}{params[1][1]}_{"toy3ssr100scale"}_abp_{params[0]} '
        # f' --lr 0.001 --bp_layer 2  --aug 1  --c0 {params[1][0]} --c1 {params[1][1]}\n'

        f'python train_cnn01_01.py --nrows {200 / 2000} --localit 1 '
        f'--updated_conv_features 32 --updated_fc_features 128 --updated_fc_nodes 1 '
        f'--updated_conv_nodes 1 --width 100 --normalize 0 '
        f'--percentile 1 --fail_count 1 --loss 01loss --act sign '
        f'--fc_diversity 1 --init normal --no_bias 2 --scale 1 '
        f'--w-inc1 {0.025} --w-inc2 {0.1} --version {"toy4srr100scale"} --seed {params[0]} --iters 5000 '
        f'--dataset {params[2][0]} --n_classes 2 --cnn 1 --divmean 0 --batch-size 128  '
        f'--target {params[2][0]}_{params[1][0]}{params[1][1]}_{"toy4srr100scale"}_abp_{params[0]} '
        f'--updated_fc_ratio 1 --updated_conv_ratio 2 --verbose_iter 500 --freeze_layer 0 '
        f'--lr 0.001 --bp_layer 5 --aug 1 --c0 {params[1][0]} --c1 {params[1][1]} --n-jobs 4 \n'
        #
        f'python train_cnn01_01.py --nrows {200 / 2000} --localit 1 '
        f'--updated_conv_features 32 --updated_fc_features 128 --updated_fc_nodes 1 '
        f'--updated_conv_nodes 1 --width 100 --normalize 0 '
        f'--percentile 1 --fail_count 1 --loss 01loss --act sign '
        f'--fc_diversity 1 --init normal --no_bias 2 --scale 1 '
        f'--w-inc1 {0.025} --w-inc2 {0.05} --version {"toy4ssr100scale"} --seed {params[0]} --iters 5000 '
        f'--dataset {params[2][0]} --n_classes 2 --cnn 1 --divmean 0 --reinit 4 --batch-size 128  '
        f'--target {params[2][0]}_{params[1][0]}{params[1][1]}_{"toy4ssr100scale"}_abp_{params[0]} '
        f'--updated_fc_ratio 1 --updated_conv_ratio 2 --verbose_iter 500 --freeze_layer 1 '
        f'--resume --source {params[2][0]}_{params[1][0]}{params[1][1]}_{"toy4srr100scale"}_abp_{params[0]} '
        f'--lr 0.001 --bp_layer 4  --aug 1  --c0 {params[1][0]} --c1 {params[1][1]} --n-jobs 4 \n'

        f'python train_cnn01_01.py --nrows {200 / 2000} --localit 1 '
        f'--updated_conv_features 32 --updated_fc_features 128 --updated_fc_nodes 1 '
        f'--updated_conv_nodes 1 --width 100 --normalize 0 '
        f'--percentile 1 --fail_count 1 --loss 01loss --act sign '
        f'--fc_diversity 1 --init normal --no_bias 2 --scale 1 '
        f'--w-inc1 {0.05} --w-inc2 {0.05} --version {"toy4sss100scale"} --seed {params[0]} --iters 5000 '
        f'--dataset {params[2][0]} --n_classes 2 --cnn 1 --divmean 0 --reinit 3 --batch-size 128 '
        f'--target {params[2][0]}_{params[1][0]}{params[1][1]}_{"toy4sss100scale"}_abp_{params[0]} '
        f'--updated_fc_ratio 1 --updated_conv_ratio 2 --verbose_iter 500 --freeze_layer 2 '
        f'--resume --source {params[2][0]}_{params[1][0]}{params[1][1]}_{"toy4ssr100scale"}_abp_{params[0]} '
        f' --lr 0.001 --bp_layer 3  --aug 1  --c0 {params[1][0]} --c1 {params[1][1]} --n-jobs 4 \n'

        f'python train_cnn01_01.py --nrows {200 / 2000} --localit 1 '
        f'--updated_conv_features 32 --updated_fc_features 128 --updated_fc_nodes 1 '
        f'--updated_conv_nodes 1 --width 100 --normalize 0 '
        f'--percentile 1 --fail_count 1 --loss 01loss --act sign '
        f'--fc_diversity 1 --init normal --no_bias 2 --scale 1 '
        f'--w-inc1 {0.05} --w-inc2 {0.1} --version {"toy4ssss100scale"} --seed {params[0]} --iters 5000 '
        f'--dataset {params[2][0]} --n_classes 2 --cnn 1 --divmean 0  --reinit 2 --batch-size 128 '
        f'--target {params[2][0]}_{params[1][0]}{params[1][1]}_{"toy4ssss100scale"}_abp_{params[0]} '
        f'--updated_fc_ratio 1 --updated_conv_ratio 2 --verbose_iter 500 --freeze_layer 3 '
        f'--resume --source {params[2][0]}_{params[1][0]}{params[1][1]}_{"toy4sss100scale"}_abp_{params[0]} '
        f' --lr 0.001 --bp_layer 2  --aug 1  --c0 {params[1][0]} --c1 {params[1][1]} --n-jobs 4 \n'
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

