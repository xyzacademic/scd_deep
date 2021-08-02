import os
import sys
import numpy as np

part_1 = [
    '#!/bin/bash -l\n',
    '#SBATCH -p datasci3,datasci4,datasci\n',
    # '#SBATCH -x node429,node430,node415\n',
    '#SBATCH --job-name=cifar10_01\n',
    '#SBATCH --gres=gpu:1\n',
    '#SBATCH --mem=16G\n',
    '#SBATCH --cpus-per-task=1\n',
    'cd ..\n',

]

part_2 = [
    'gpu=0\n',
    '\n',
]

if not os.path.exists('temp_sh'):
    os.makedirs('temp_sh')
files = []

# version = ['toy2rs100', 'toy2rs100fc2', 'toy2rs100fc3', 'toy2rs100fc3srs',
#            'toy2rs20', 'toy2rs400', 'toy2rs20fc2', 'toy2rs20fc3',
#            'toy2rs400fc2', 'toy2rs400fc3']
version = ['ag']
    # , 'toy4ssr100b','toy2s4ssr100', 'toy2s8ssr100', 'toy3s4ssr100', 'toy3s8ssr100']
seeds = np.arange(1)
# nrows = [256, 512, 768, 1024]
nrows = [50]
# acts = ['sign', 'sigmoid', 'relu']
acts = ['sign']
nbs = [2]
dms = [0]
upc = [10]
upf = [1]
ucf = [128]
lr_convs = [0.05, 0.01, 0.1, 0.2]
lr_fcs = [0.05, 0.1, 0.2]
# lr_convs = [0.5]
# lr_fcs = [0.5]

variables = []
for i in acts:
    for j in nbs:
        for k in dms:
            for o in upc:
                for p in upf:
                    for q in version:
                        for r in nrows:
                            for s in ucf:
                                for lc in lr_convs:
                                    for lf in lr_fcs:
                                        for m in seeds:
                                            # if not os.path.exists(os.path.join('../checkpoints/pt', f'cifar10_{q}test_{i}_i1_mce_b{r}nb{j}_nw0_dm{k}_upc{o}_upf{p}_ucf{s}_fp32_{m}.pt')):
                                            variables.append((i, j, k, m, o, p, q, r, s, lc, lf))

for i, params in enumerate(variables):
    # for i, (seed, target_class) in enumerate(variables):
    # for i, (votes, index) in enumerate(variables):
    # for i, (version, scale) in enumerate(variables):

    part_3 = [

        # f'python train_cnn01_word_ce.py --nrows {params[7]} --localit 1 '
        # f'--updated_conv_features {params[8]} --updated_fc_features 128 --updated_fc_nodes {params[5]} '
        # f'--updated_conv_nodes {params[4]} --width 100 --normalize 0 '
        # f'--percentile 1 --fail_count 1 --loss mce --act {params[0]} '
        # f'--fc_diversity 1 --init normal --no_bias {params[1]} --scale 1 '
        # f'--w-inc1 {0.05} --w-inc2 {0.05} --version {"wordcnn01"} --seed {params[3]} --iters 50 '
        # f'--dataset mr --n_classes 2 --cnn 1 --divmean {params[2]} '
        # # f'--target sms_{"wordcnn01"}_bp_{params[0]}_i1_mce_b{params[7]}_lrc{0.05}_lrf{0.05}_nb{params[1]}_nw0_dm{params[2]}_upc{params[4]}_upf{params[5]}_ucf{params[8]}_{params[3]} '
        # f'--target mr_{"wordcnn01"}_bp_01_fs_{params[3]} '
        # f'--updated_fc_ratio 5 --updated_conv_ratio 5 --verbose_iter 1 --freeze_layer 0 '
        # f'--resume --source mr_bp_d0_encoder_{params[3]} '
        # f'--dropout 0.0 \n'

        # f'python train_cnn01_word_ce.py --nrows {params[7]} --localit 1 '
        # f'--updated_conv_features {params[8]} --updated_fc_features 128 --updated_fc_nodes {params[5]} '
        # f'--updated_conv_nodes {params[4]} --width 100 --normalize 0 '
        # f'--percentile 1 --fail_count 1 --loss mce --act {params[0]} '
        # f'--fc_diversity 1 --init normal --no_bias {params[1]} --scale 1 '
        # f'--w-inc1 {0.05} --w-inc2 {0.05} --version {"wordcnn01"} --seed {params[3]} --iters 50 '
        # f'--dataset sms --n_classes 2 --cnn 1 --divmean {params[2]} '
        # f'--target sms_{"wordcnn01"}_bp_long_01_fs_{params[3]} '
        # f'--updated_fc_ratio 5 --updated_conv_ratio 5 --verbose_iter 1 --freeze_layer 0 '
        # f'--resume --source sms_bp_d0_long_encoder_{params[3]} '
        # f'--dropout 0.0 \n'

        # f'python train_cnn01_word_ce.py --nrows {params[7]} --localit 1 '
        # f'--updated_conv_features {params[8]} --updated_fc_features 128 --updated_fc_nodes {params[5]} '
        # f'--updated_conv_nodes {params[4]} --width 100 --normalize 0 '
        # f'--percentile 1 --fail_count 1 --loss mce --act {params[0]} '
        # f'--fc_diversity 1 --init normal --no_bias {params[1]} --scale 1 '
        # f'--w-inc1 {0.05} --w-inc2 {0.05} --version {"wordcnn01"} --seed {params[3]} --iters 50 '
        # f'--dataset ag --n_classes 4 --cnn 1 --divmean {params[2]} '
        # f'--target ag_{"wordcnn01"}_bp_01_{params[3]} '
        # f'--updated_fc_ratio 5 --updated_conv_ratio 5 --verbose_iter 1 --freeze_layer 0 '
        # f'--resume --source ag_bp_d0_encoder_{params[3]} '
        # f'--dropout 0.0 \n'

        # f'python train_cnn01_word_ce.py --nrows {params[7]} --localit 1 '
        # f'--updated_conv_features {params[8]} --updated_fc_features 128 --updated_fc_nodes {params[5]} '
        # f'--updated_conv_nodes {params[4]} --width 100 --normalize 0 '
        # f'--percentile 1 --fail_count 1 --loss mce --act {params[0]} '
        # f'--fc_diversity 1 --init normal --no_bias {params[1]} --scale 1 '
        # f'--w-inc1 {0.00} --w-inc2 {0.00} --version {"wordcnn01"} --seed {params[3]} --iters 0 '
        # f'--dataset ag --n_classes 4 --cnn 1 --divmean {params[2]} '
        # f'--target ag_{"wordcnn01"}_bp_relu_{params[3]} '
        # f'--updated_fc_ratio 5 --updated_conv_ratio 5 --verbose_iter 1 --freeze_layer 0 '
        # f'--resume --source ag_bp_d0_encoder_{params[3]} '
        # f'--dropout 0.0 \n'

        # f'python train_cnn01_word_ce.py --nrows {params[7]} --localit 1 '
        # f'--updated_conv_features {params[8]} --updated_fc_features 128 --updated_fc_nodes {params[5]} '
        # f'--updated_conv_nodes {params[4]} --width 100 --normalize 0 '
        # f'--percentile 1 --fail_count 1 --loss mce --act {params[0]} '
        # f'--fc_diversity 1 --init normal --no_bias {params[1]} --scale 1 '
        # f'--w-inc1 {0.00} --w-inc2 {0.00} --version {"wordcnn01"} --seed {params[3]} --iters 0 '
        # f'--dataset imdb --n_classes 2 --cnn 1 --divmean {params[2]} '
        # f'--target imdb_{"wordcnn01"}_bp_relu_fs_{params[3]} '
        # f'--updated_fc_ratio 5 --updated_conv_ratio 5 --verbose_iter 1 --freeze_layer 0 '
        # f'--resume --source imdb_bp_d0_encoder_{params[3]} '
        # f'--dropout 0.0 \n'

        # f'python train_cnn01_word_ce.py --nrows {params[7]} --localit 1 '
        # f'--updated_conv_features {params[8]} --updated_fc_features 128 --updated_fc_nodes {params[5]} '
        # f'--updated_conv_nodes {params[4]} --width 100 --normalize 0 '
        # f'--percentile 1 --fail_count 1 --loss mce --act {params[0]} '
        # f'--fc_diversity 1 --init normal --no_bias {params[1]} --scale 1 '
        # f'--w-inc1 {0.05} --w-inc2 {0.05} --version {"wordcnn01"} --seed {params[3]} --iters 50 '
        # f'--dataset imdb --n_classes 2 --cnn 1 --divmean {params[2]} '
        # f'--target imdb_{"wordcnn01"}_bp_01_fs_{params[3]} '
        # f'--updated_fc_ratio 5 --updated_conv_ratio 5 --verbose_iter 1 --freeze_layer 0 '
        # f'--resume --source imdb_bp_d0_encoder_{params[3]} '
        # f'--dropout 0.0 \n'

        # f'python train_cnn01_word_ce.py --nrows {params[7]} --localit 1 '
        # f'--updated_conv_features {params[8]} --updated_fc_features 128 --updated_fc_nodes {params[5]} '
        # f'--updated_conv_nodes {params[4]} --width 100 --normalize 0 '
        # f'--percentile 1 --fail_count 1 --loss mce --act {params[0]} '
        # f'--fc_diversity 1 --init normal --no_bias {params[1]} --scale 1 '
        # f'--w-inc1 {0.00} --w-inc2 {0.00} --version {"wordcnn01"} --seed {params[3]} --iters 0 '
        # f'--dataset yelp --n_classes 2 --cnn 1 --divmean {params[2]} '
        # f'--target yelp_{"wordcnn01"}_bp_relu_{params[3]} '
        # f'--updated_fc_ratio 5 --updated_conv_ratio 5 --verbose_iter 1 --freeze_layer 0 '
        # f'--resume --source yelp_bp_d0_encoder_{params[3]} '
        # f'--dropout 0.0 \n'

        # f'python train_cnn01_word_ce.py --nrows {params[7]} --localit 1 '
        # f'--updated_conv_features {params[8]} --updated_fc_features 128 --updated_fc_nodes {params[5]} '
        # f'--updated_conv_nodes {params[4]} --width 100 --normalize 0 '
        # f'--percentile 1 --fail_count 1 --loss mce --act {params[0]} '
        # f'--fc_diversity 1 --init normal --no_bias {params[1]} --scale 1 '
        # f'--w-inc1 {0.05} --w-inc2 {0.05} --version {"wordcnn01"} --seed {params[3]} --iters 50 '
        # f'--dataset yelp --n_classes 2 --cnn 1 --divmean {params[2]} '
        # f'--target yelp_{"wordcnn01"}_bp_01_fs_{params[3]} '
        # f'--updated_fc_ratio 5 --updated_conv_ratio 5 --verbose_iter 1 --freeze_layer 0 '
        # f'--resume --source yelp_bp_d0_encoder_{params[3]} '
        # f'--dropout 0.0 \n'

        # f'python train_cnn01_word_ce.py --nrows {params[7]} --localit 1 '
        # f'--updated_conv_features {params[8]} --updated_fc_features 128 --updated_fc_nodes {params[5]} '
        # f'--updated_conv_nodes {params[4]} --width 100 --normalize 0 '
        # f'--percentile 1 --fail_count 1 --loss mce --act {params[0]} '
        # f'--fc_diversity 1 --init normal --no_bias {params[1]} --scale 1 '
        # f'--w-inc1 {0.05} --w-inc2 {0.05} --version {"wordcnn01"} --seed {params[3]} --iters 50 '
        # f'--dataset {params[6]} --n_classes 2 --cnn 1 --divmean {params[2]} '
        # f'--target {params[6]}_{"wordcnn01"}_bp_{params[0]}_i1_mce_b{params[7]}_lrc{0.05}_lrf{0.05}_nb{params[1]}_nw0_dm{params[2]}_upc{params[4]}_upf{params[5]}_ucf{params[8]}_{params[3]} '
        # # f'--target {params[6]}_{"wordcnn01"}_bp_01_fs_{params[3]} '
        # f'--updated_fc_ratio 5 --updated_conv_ratio 5 --verbose_iter 1 --freeze_layer 0 '
        # f'--resume --source {params[6]}_bp_d0_encoder_{params[3]} '
        # f'--dropout 0.0 \n'

        f'python train_cnn01_word_ce.py --nrows {params[7]} --localit 1 '
        f'--updated_conv_features {params[8]} --updated_fc_features 128 --updated_fc_nodes {params[5]} '
        f'--updated_conv_nodes {params[4]} --width 100 --normalize 0 '
        f'--percentile 1 --fail_count 1 --loss mce --act {params[0]} '
        f'--fc_diversity 1 --init normal --no_bias {params[1]} --scale 1 '
        f'--w-inc1 {params[9]} --w-inc2 {params[10]} --version {"wordcnn01"} --seed {params[3]} --iters 50 '
        f'--dataset {params[6]} --n_classes 4 --cnn 1 --divmean {params[2]} '
        f'--target {params[6]}_{"wordcnn01"}_fs_bp_{params[0]}_i1_mce_b{params[7]}_lrc{params[9]}_lrf{params[10]}_nb{params[1]}_nw0_dm{params[2]}_upc{params[4]}_upf{params[5]}_ucf{params[8]}_{params[3]} '
        # f'--target {params[6]}_{"wordcnn01"}_bp_01_fs_{params[3]} '
        f'--updated_fc_ratio 5 --updated_conv_ratio 5 --verbose_iter 1 --freeze_layer 0 '
        f'--resume --source {params[6]}_bp_d0_encoder_{params[3]} '
        f'--dropout 0.0 \n'
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

