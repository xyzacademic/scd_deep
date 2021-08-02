#!/bin/sh

#$-q datasci
#$-q datasci3
#$-cwd
#$-N scd_attack

cd ..

gpu=1

for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
do

python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target stl10_scd01_single_vote --random-sign 1 --seed $seed --dataset stl10 --oracle-size 1024

python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target stl10_scd01_100_vote --random-sign 1 --seed $seed --dataset stl10 --oracle-size 1024

python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target stl10_scd01mlp_single_vote --random-sign 1 --seed $seed --dataset stl10 --oracle-size 1024

python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target stl10_scd01mlp_32_vote --random-sign 1 --seed $seed --dataset stl10 --oracle-size 1024

python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target stl10_scd01_v12_adv --random-sign 1 --seed $seed --dataset stl10 --oracle-size 1024

python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target stl10_scd01_v13_adv --random-sign 1 --seed $seed --dataset stl10 --oracle-size 1024

python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target stl10_scd01mlp_v12_adv --random-sign 1 --seed $seed --dataset stl10 --oracle-size 1024

python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target stl10_scd01mlp_v13_adv --random-sign 1 --seed $seed --dataset stl10 --oracle-size 1024

python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target stl10_scd01_v16_adv --random-sign 1 --seed $seed --dataset stl10 --oracle-size 1024

python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target stl10_scd01_v17_adv --random-sign 1 --seed $seed --dataset stl10 --oracle-size 1024

python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target stl10_scd01mlp_v16_adv --random-sign 1 --seed $seed --dataset stl10 --oracle-size 1024

python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target stl10_scd01mlp_v17_adv --random-sign 1 --seed $seed --dataset stl10 --oracle-size 1024

python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target stl10_svm --random-sign 1 --seed $seed --dataset stl10 --oracle-size 1024

python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target stl10_mlp --random-sign 1 --seed $seed --dataset stl10 --oracle-size 1024

done
