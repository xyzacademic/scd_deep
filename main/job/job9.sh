#!/bin/sh

#$-q datasci
#$-q datasci3
#$-q datasci4
#$-cwd
#$-N imagenet

cd ..




python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 1000 \
--updated-features 128 --round 1 --interval 20 --n-jobs 4 --num-gpus 4 --save \
--target imagenet_scd01_single_vote.pkl --dataset imagenet --version v1 --seed 2018 >> imagenet_logs

python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 1000 \
--updated-features 128 --round 100 --interval 20 --n-jobs 4 --num-gpus 4 --save \
--target imagenet_scd01_100_vote.pkl --dataset imagenet --version v1 --seed 2018 >> imagenet_logs

python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
--updated-features 128 --round 1 --interval 20 --n-jobs 4 --num-gpus 4 --save \
--target imagenet_scd01mlp_single_vote.pkl --dataset imagenet --version v2 --seed 2018 >> imagenet_logs

python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
--updated-features 128 --round 32 --interval 20 --n-jobs 4 --num-gpus 4 --save \
--target imagenet_scd01mlp_32_vote.pkl --dataset imagenet --version v2 --seed 2018 >> imagenet_logs

python train_svm.py --dataset imagenet --c 0.01 --target imagenet_svm.pkl --save

python train_mlp.py --dataset imagenet --hidden-nodes 20 --lr 0.01 --iters 1000 --target imagenet_mlp.pkl --seed 2018 --save

cd job

qsub -l hostname=node438 imagenet_blackbox_attack.sh