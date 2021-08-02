import numpy as np
import os


model_list = [
    'toy3rrr_bp_fp16_32',
    'cifar10_toy3rsss100_adaptivebs_bp_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32',
    'cifar10_toy3rrss100_adaptivebs_bp_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32',
    'cifar10_toy3ssss100_adaptivebs_bp_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32',
    'cifar10_toy3bnn_dm1_approx',
]

indices = np.load('cifar10_interection_1000.npy')

os.chdir('adv_data/20')

for model in model_list:
    data = np.concatenate([np.load(os.path.join(model, f'{index}.npy')) for index in indices], axis=0)
    print('data shape: ', data.shape)
    np.save(f'{model}_adv.npy', data)
    del data
