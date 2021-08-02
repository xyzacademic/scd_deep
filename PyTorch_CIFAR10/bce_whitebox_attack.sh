#!/bin/sh

#target_model="
#cifar10_binary_toy3rrr_nb2_bce_bp_8
#cifar10_binary_toy3srr100scale_nb2_bce_bp_8
#cifar10_binary_toy3ssr100scale_nb2_bce_bp_8
#cifar10_binary_toy3sss100scale_nb2_bce_bp_8
#cifar10_binary_toy3ssss100scale_nb2_bce_bp_8
#cifar10_binary_toy3srr100scale_abp_sign_i1_bce_b200_lrc0.025_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8
#cifar10_binary_toy3ssr100scale_abp_sign_i1_bce_b200_lrc0.025_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8
#cifar10_binary_toy3sss100scale_abp_sign_i1_bce_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8
#cifar10_binary_toy3ssss100scale_abp_sign_i1_bce_b200_lrc0.05_lrf0.7_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8
#cifar10_binary_toy3srr100scale_abp_sign_i1_01loss_b200_lrc0.025_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8
#cifar10_binary_toy3ssr100scale_abp_sign_i1_01loss_b200_lrc0.025_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8
#cifar10_binary_toy3sss100scale_abp_sign_i1_01loss_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8
#cifar10_binary_toy3ssss100scale_abp_sign_i1_01loss_b200_lrc0.05_lrf0.7_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8
#"

#target_model="
#cifar10_binary_toy3rrr100_nb2_bce_adv_bp_8
#cifar10_binary_toy3srr100scale_nb2_bce_adv_bp_8
#cifar10_binary_toy3ssr100scale_nb2_bce_adv_bp_8
#cifar10_binary_toy3sss100scale_nb2_bce_adv_bp_8
#cifar10_binary_toy3ssss100scale_nb2_bce_adv_bp_8
#cifar10_binary_toy3srr100scale_abp_adv_sign_i1_bce_b200_lrc0.025_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8
#cifar10_binary_toy3ssr100scale_abp_adv_sign_i1_bce_b200_lrc0.025_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8
#cifar10_binary_toy3sss100scale_abp_adv_sign_i1_bce_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8
#cifar10_binary_toy3ssss100scale_abp_adv_sign_i1_bce_b200_lrc0.05_lrf0.7_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8
#cifar10_binary_toy3srr100scale_abp_adv_sign_i1_01loss_b200_lrc0.025_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8
#cifar10_binary_toy3ssr100scale_abp_adv_sign_i1_01loss_b200_lrc0.025_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8
#cifar10_binary_toy3sss100scale_abp_adv_sign_i1_01loss_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8
#cifar10_binary_toy3ssss100scale_abp_adv_sign_i1_01loss_b200_lrc0.05_lrf0.7_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8
#"

target_model="
cifar10_binary_toy3rrr_nb2_bce_bp
cifar10_binary_toy3srr100scale_nb2_bce_bp
cifar10_binary_toy3ssr100scale_nb2_bce_bp
cifar10_binary_toy3sss100scale_nb2_bce_bp
cifar10_binary_toy3ssss100scale_nb2_bce_bp
cifar10_binary_toy3srr100scale_abp_sign_i1_bce_b200_lrc0.025_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal
cifar10_binary_toy3ssr100scale_abp_sign_i1_bce_b200_lrc0.025_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal
cifar10_binary_toy3sss100scale_abp_sign_i1_bce_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal
cifar10_binary_toy3ssss100scale_abp_sign_i1_bce_b200_lrc0.05_lrf0.7_nb2_nw0_dm0_upc1_upf1_ucf32_normal
cifar10_binary_toy3srr100scale_abp_sign_i1_01loss_b200_lrc0.025_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal
cifar10_binary_toy3ssr100scale_abp_sign_i1_01loss_b200_lrc0.025_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal
cifar10_binary_toy3sss100scale_abp_sign_i1_01loss_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal
cifar10_binary_toy3ssss100scale_abp_sign_i1_01loss_b200_lrc0.05_lrf0.7_nb2_nw0_dm0_upc1_upf1_ucf32_normal
cifar10_binary_toy3rrr100_nb2_bce_adv_bp
cifar10_binary_toy3srr100scale_nb2_bce_adv_bp
cifar10_binary_toy3ssr100scale_nb2_bce_adv_bp
cifar10_binary_toy3sss100scale_nb2_bce_adv_bp
cifar10_binary_toy3ssss100scale_nb2_bce_adv_bp
cifar10_binary_toy3srr100scale_abp_adv_sign_i1_bce_b200_lrc0.025_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal
cifar10_binary_toy3ssr100scale_abp_adv_sign_i1_bce_b200_lrc0.025_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal
cifar10_binary_toy3sss100scale_abp_adv_sign_i1_bce_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal
cifar10_binary_toy3ssss100scale_abp_adv_sign_i1_bce_b200_lrc0.05_lrf0.7_nb2_nw0_dm0_upc1_upf1_ucf32_normal
cifar10_binary_toy3srr100scale_abp_adv_sign_i1_01loss_b200_lrc0.025_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal
cifar10_binary_toy3ssr100scale_abp_adv_sign_i1_01loss_b200_lrc0.025_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal
cifar10_binary_toy3sss100scale_abp_adv_sign_i1_01loss_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal
cifar10_binary_toy3ssss100scale_abp_adv_sign_i1_01loss_b200_lrc0.05_lrf0.7_nb2_nw0_dm0_upc1_upf1_ucf32_normal
"
for target in $target_model:
do
python bce_attack_mx.py --epsilon 16 --num-steps 10 --attack_type pgd --n_classes 2 --source ${target} --name results/0220/${target}
done

#for target in $target_model:
#do
#python bce_attack.py --epsilon 16 --num-steps 10 --attack_type mia --n_classes 2 --source $target >> mia_whitebox_0219_thm4_eps16
#done

python mlp_attack_out.py --epsilon 16 --num-steps 20 --attack_type pgd --dataset cifar10_binary --c0 2 --c1 3 --name results/cifar10_binary/mlp_16_20_23
python mlp_attack_in.py --epsilon 16 --num-steps 20 --attack_type pgd --dataset cifar10_binary --c0 2 --c1 3 --votes 8 --target mlprr --source mlpbcebp --name results/inner/cifar10_binary/mlpbcebp_16_20_23

python cnn_attack_out.py --epsilon 16 --num-steps 20 --attack_type pgd --dataset cifar10_binary --c0 2 --c1 3 --name results/cnn_16_20_23 --cnn 1