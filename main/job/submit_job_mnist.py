import os
import sys




part_1 = [
    '#!/bin/bash -l\n',
    '#SBATCH -p datasci\n',
    '#SBATCH --job-name=mnist_job\n',
    '#SBATCH --gres=gpu:1\n',
    '#SBATCH --mem=16G\n',

    'cd ../\n',

]

part_2 = [
    'gpu=0\n',
    '\n',
]

files = []

for i in range(10):
    variable_key = 2 * (i+1)
    variable_1 = variable_key
    variable_2 = variable_key / 10

    part_3 = [

        'python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.05 --num-iters 100 --b-ratio 0.2 \
        --updated-features 64 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
        --target mnist_hinge_%d.pkl --dataset mnist  --seed 2018 --c %.1f\n' % (variable_1, variable_2),

        'for seed in 2019 2393 92382 232 12 58 954 758 451 2015687\n',

        'do\n',

        'python bb_attack.py --epsilon 0.3 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 \
        --train-size 200 --target mnist_hinge_%d --random-sign 1 --seed $seed --dataset mnist \
        --oracle-size 1024 --n_classes 2\n' % (variable_1),

        'done\n',

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

