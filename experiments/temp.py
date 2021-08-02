import os

# targets = ['cifar10_binary_toy3rrr_nb2_bce_bp',
# 'cifar10_binary_toy3srr100scale_nb2_bce_bp',
# 'cifar10_binary_toy3ssr100scale_nb2_bce_bp',
# 'cifar10_binary_toy3sss100scale_nb2_bce_bp',
# 'cifar10_binary_toy3ssss100scale_nb2_bce_bp',
# 'cifar10_binary_toy3srr100scale_abp_sign_i1_bce_b200_lrc0.025_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal',
# 'cifar10_binary_toy3ssr100scale_abp_sign_i1_bce_b200_lrc0.025_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal',
# 'cifar10_binary_toy3sss100scale_abp_sign_i1_bce_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal',
# 'cifar10_binary_toy3ssss100scale_abp_sign_i1_bce_b200_lrc0.05_lrf0.7_nb2_nw0_dm0_upc1_upf1_ucf32_normal',
# 'cifar10_binary_toy3srr100scale_abp_sign_i1_01loss_b200_lrc0.025_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal',
# 'cifar10_binary_toy3ssr100scale_abp_sign_i1_01loss_b200_lrc0.025_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal',
# 'cifar10_binary_toy3sss100scale_abp_sign_i1_01loss_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal',
# 'cifar10_binary_toy3ssss100scale_abp_sign_i1_01loss_b200_lrc0.05_lrf0.7_nb2_nw0_dm0_upc1_upf1_ucf32_normal',
#
# 'cifar10_binary_toy3rrr100_nb2_bce_adv_bp',
# 'cifar10_binary_toy3srr100scale_nb2_bce_adv_bp',
# 'cifar10_binary_toy3ssr100scale_nb2_bce_adv_bp',
# 'cifar10_binary_toy3sss100scale_nb2_bce_adv_bp',
# 'cifar10_binary_toy3ssss100scale_nb2_bce_adv_bp',
# 'cifar10_binary_toy3srr100scale_abp_adv_sign_i1_bce_b200_lrc0.025_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal',
# 'cifar10_binary_toy3ssr100scale_abp_adv_sign_i1_bce_b200_lrc0.025_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal',
# 'cifar10_binary_toy3sss100scale_abp_adv_sign_i1_bce_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal',
# 'cifar10_binary_toy3ssss100scale_abp_adv_sign_i1_bce_b200_lrc0.05_lrf0.7_nb2_nw0_dm0_upc1_upf1_ucf32_normal',
# 'cifar10_binary_toy3srr100scale_abp_adv_sign_i1_01loss_b200_lrc0.025_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal',
# 'cifar10_binary_toy3ssr100scale_abp_adv_sign_i1_01loss_b200_lrc0.025_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal',
# 'cifar10_binary_toy3sss100scale_abp_adv_sign_i1_01loss_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal',
# 'cifar10_binary_toy3ssss100scale_abp_adv_sign_i1_01loss_b200_lrc0.05_lrf0.7_nb2_nw0_dm0_upc1_upf1_ucf32_normal',
# ]
#
# versions = ['toy3rrr100',
# 'toy3srr100scale',
# 'toy3ssr100scale',
# 'toy3sss100scale',
# 'toy3ssss100scale',
# 'toy3srr100scale',
# 'toy3ssr100scale',
# 'toy3sss100scale',
# 'toy3ssss100scale',
# 'toy3srr100scale',
# 'toy3ssr100scale',
# 'toy3sss100scale',
# 'toy3ssss100scale',
#
# 'toy3rrr100',
# 'toy3srr100scale',
# 'toy3ssr100scale',
# 'toy3sss100scale',
# 'toy3ssss100scale',
# 'toy3srr100scale',
# 'toy3ssr100scale',
# 'toy3sss100scale',
# 'toy3ssss100scale',
# 'toy3srr100scale',
# 'toy3ssr100scale',
# 'toy3sss100scale',
# 'toy3ssss100scale',
# ]
#
# with open('votes.sh', 'w') as f:
#     f.write('#!/bin/sh\n\n')
#
#     for i, (version, target) in enumerate(zip(versions, targets)):
#         for seed in range(8):
#             f.write(f'python combine_vote_mlp.py --dataset cifar10 '
#                 f'--n_classes 2 --votes 1 --no_bias 1 '
#                 f'--scale 1 --cnn 1 --version {version} --act sign '
#                 f'--target {target} --save --seed {seed} \n')


targets = [
    'toy3rrr_bp_fp16',
    'cifar10_toy3srr100scale_bp_fp16',
    'cifar10_toy3ssr100scale_bp_fp16',
    'cifar10_toy3sss100scale_bp_fp16',
    'cifar10_toy3ssss100scale_bp_fp16',

    'cifar10_toy3srr100scale_abp_sign_i1_mce_b200_lrc0.025_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16',
    'cifar10_toy3ssr100scale_abp_retrain0_sign_i1_mce_b200_lrc0.025_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16',
    'cifar10_toy3sss100scale_abp_retrain0_sign_i1_mce_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16',
    'cifar10_toy3ssss100scale_abp_retrain0_sign_i1_mce_b200_lrc0.05_lrf0.7_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16',

    'cifar10_toy3srr100scale_abp_sign_i1_01lossmc_b200_lrc0.025_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16',
    'cifar10_toy3ssr100scale_abp_sign_i1_01lossmc_b200_lrc0.025_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16',
    'cifar10_toy3sss100scale_abp_sign_i1_01lossmc_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16',
    'cifar10_toy3ssss100scale_abp_sign_i1_01lossmc_b200_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16',
]

versions = [
'toy3rrr100',
'toy3srr100scale',
'toy3ssr100scale',
'toy3sss100scale',
'toy3ssss100scale',

'toy3srr100scale',
'toy3ssr100scale',
'toy3sss100scale',
'toy3ssss100scale',

'toy3srr100scale',
'toy3ssr100scale',
'toy3sss100scale',
'toy3ssss100scale',

]

with open('votes.sh', 'w') as f:
    f.write('#!/bin/sh\n\n')

    for i, (version, target) in enumerate(zip(versions, targets)):
        for seed in range(8):
            f.write(f'python combine_vote_new.py --dataset cifar10 '
                f'--n_classes 10 --votes 1 --no_bias 1 '
                f'--scale 1 --cnn 1 --version {version} --act sign '
                f'--target {target} --save --seed {seed} \n')