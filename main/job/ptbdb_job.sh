#!/bin/bash -l

#SBATCH -p datasci3

#SBATCH --job-name=ptbdb_job
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

cd ..


python train_scd4.py --nrows 0.15 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
--b-ratio 0.2 --updated-features 128 --round 8 --interval 20 --n-jobs 4 --num-gpus 4  --n_classes 2 \
--iters 1 --target ptbdb_scd01mlpbnn_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset ptbdb --version cebnn --seed 0 \
--width 10000 --metrics balanced --init normal  --updated-nodes 1

python train_scd4.py --nrows 0.15 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 3000 \
--b-ratio 0.2 --updated-features 128 --round 8 --interval 20 --n-jobs 4 --num-gpus 4  --n_classes 2 \
--iters 1 --target ptbdb_scd01mlpbnn_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset ptbdb --version cebnn --seed 0 \
--width 10000 --metrics balanced --init normal  --updated-nodes 1

python train_scd4.py --nrows 0.15 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 5000 \
--b-ratio 0.2 --updated-features 128 --round 8 --interval 20 --n-jobs 4 --num-gpus 4  --n_classes 2 \
--iters 1 --target ptbdb_scd01mlpbnn_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset ptbdb --version cebnn --seed 0 \
--width 10000 --metrics balanced --init normal  --updated-nodes 1

python train_scd4.py --nrows 0.15 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 8000 \
--b-ratio 0.2 --updated-features 128 --round 8 --interval 20 --n-jobs 4 --num-gpus 4  --n_classes 2 \
--iters 1 --target ptbdb_scd01mlpbnn_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset ptbdb --version cebnn --seed 0 \
--width 10000 --metrics balanced --init normal  --updated-nodes 1

python train_scd4.py --nrows 0.15 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 10000 \
--b-ratio 0.2 --updated-features 128 --round 8 --interval 20 --n-jobs 4 --num-gpus 4  --n_classes 2 \
--iters 1 --target ptbdb_scd01mlpbnn_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset ptbdb --version cebnn --seed 0 \
--width 10000 --metrics balanced --init normal  --updated-nodes 1


python train_scd4.py --nrows 0.25 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
--b-ratio 0.2 --updated-features 128 --round 8 --interval 20 --n-jobs 4 --num-gpus 4  --n_classes 2 \
--iters 1 --target ptbdb_scd01mlpbnn_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset ptbdb --version cebnn --seed 0 \
--width 10000 --metrics balanced --init normal  --updated-nodes 1

python train_scd4.py --nrows 0.25 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 3000 \
--b-ratio 0.2 --updated-features 128 --round 8 --interval 20 --n-jobs 4 --num-gpus 4  --n_classes 2 \
--iters 1 --target ptbdb_scd01mlpbnn_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset ptbdb --version cebnn --seed 0 \
--width 10000 --metrics balanced --init normal  --updated-nodes 1

python train_scd4.py --nrows 0.25 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 5000 \
--b-ratio 0.2 --updated-features 128 --round 8 --interval 20 --n-jobs 4 --num-gpus 4  --n_classes 2 \
--iters 1 --target ptbdb_scd01mlpbnn_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset ptbdb --version cebnn --seed 0 \
--width 10000 --metrics balanced --init normal  --updated-nodes 1

python train_scd4.py --nrows 0.25 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 8000 \
--b-ratio 0.2 --updated-features 128 --round 8 --interval 20 --n-jobs 4 --num-gpus 4  --n_classes 2 \
--iters 1 --target ptbdb_scd01mlpbnn_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset ptbdb --version cebnn --seed 0 \
--width 10000 --metrics balanced --init normal  --updated-nodes 1

python train_scd4.py --nrows 0.25 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 10000 \
--b-ratio 0.2 --updated-features 128 --round 8 --interval 20 --n-jobs 4 --num-gpus 4  --n_classes 2 \
--iters 1 --target ptbdb_scd01mlpbnn_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset ptbdb --version cebnn --seed 0 \
--width 10000 --metrics balanced --init normal  --updated-nodes 1


python train_scd4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
--b-ratio 0.2 --updated-features 128 --round 8 --interval 20 --n-jobs 4 --num-gpus 4  --n_classes 2 \
--iters 1 --target ptbdb_scd01mlpbnn_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset ptbdb --version cebnn --seed 0 \
--width 10000 --metrics balanced --init normal  --updated-nodes 1

python train_scd4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 3000 \
--b-ratio 0.2 --updated-features 128 --round 8 --interval 20 --n-jobs 4 --num-gpus 4  --n_classes 2 \
--iters 1 --target ptbdb_scd01mlpbnn_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset ptbdb --version cebnn --seed 0 \
--width 10000 --metrics balanced --init normal  --updated-nodes 1

python train_scd4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 5000 \
--b-ratio 0.2 --updated-features 128 --round 8 --interval 20 --n-jobs 4 --num-gpus 4  --n_classes 2 \
--iters 1 --target ptbdb_scd01mlpbnn_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset ptbdb --version cebnn --seed 0 \
--width 10000 --metrics balanced --init normal  --updated-nodes 1

python train_scd4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 8000 \
--b-ratio 0.2 --updated-features 128 --round 8 --interval 20 --n-jobs 4 --num-gpus 4  --n_classes 2 \
--iters 1 --target ptbdb_scd01mlpbnn_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset ptbdb --version cebnn --seed 0 \
--width 10000 --metrics balanced --init normal  --updated-nodes 1

python train_scd4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 10000 \
--b-ratio 0.2 --updated-features 128 --round 8 --interval 20 --n-jobs 4 --num-gpus 4  --n_classes 2 \
--iters 1 --target ptbdb_scd01mlpbnn_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset ptbdb --version cebnn --seed 0 \
--width 10000 --metrics balanced --init normal  --updated-nodes 1

