import sys
sys.path.append('..')
import pickle
import os
from tools import args, save_checkpoint, print_title, load_data
import numpy as np
from art.attacks.evasion import HopSkipJump
from art.classifiers import BlackBoxClassifier
import time
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

train, test, train_label, test_label = load_data('cifar10', 10)

train = train.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2)).astype(np.float32)
test = test.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2)).astype(np.float32)


source_model = [    
    'toy3rrr_bp_fp16_32',
    'cifar10_toy3rsss100_adaptivebs_bp_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32',
    'cifar10_toy3rrss100_adaptivebs_bp_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32',
    'cifar10_toy3ssss100_adaptivebs_bp_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32',
    'cifar10_toy3bnn_dm1_approx',
    # 'cifar10_toy3rrr100_bp_adv_32',
    # 'cifar10_toy3rrss100_adaptivebs_bp_adv_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32',
    # 'cifar10_toy3rsss100_adaptivebs_bp_adv_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32',
    # 'cifar10_toy3ssss100_adaptivebs_bp_adv_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32',
    # 'cifar10_toy3rrss100_adaptivebs_bp_sign_i1_01loss_b1000_lrc0.05_lrf0.01_nb2_nw0_dm0_upc1_upf10_ucf32_fp16_32',
    # 'cifar10_toy3rsss100_bp_sign_i1_01loss_b5000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf10_ucf32_fp16_32',
    # 'cifar10_toy3ssss100_bp_sign_i1_01loss_b5000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf10_ucf32_fp16_32',
    # 'cifar10_toy3rrss100_bp_sign_i1_01loss_b5000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf10_ucf32_fp16_32',

]

target_model = source_model

# index = [1760, 1716, 3480, 3628, 2961, 1117, 7466, 2791, 2502, 5332,
#          1651, 351, 9234, 5033, 9074, 319, 777, 8074, 3950, 3979]
index = np.load('cifar10_interection_1000.npy')
label = test_label[index]



model_yp = {}
for model in target_model:
    model_yp[model] = {}


for model in target_model:
    if 'approx' in model:
        from core.bnn import BNN
        scd = BNN(['checkpoints/%s_%d.h5' % (model, i) for i in range(32)])
    else:
        with open(f'checkpoints/{model}.pkl', 'rb') as f:
            scd = pickle.load(f)

    for source in source_model:
        adv_data = np.load(f'adv_data/20/{source}_adv.npy')
        if 'approx' in model:
            yp = scd.predict(adv_data*2 - 1)
        else:
            yp = scd.predict(adv_data)
        model_yp[model][source] = yp

    del scd, yp


acc_df = pd.DataFrame(columns=['source/target'] + source_model)
match_df = pd.DataFrame(columns=['source/target'] + source_model)
acc_df['source/target'] = source_model
match_df['source/target'] = source_model

for i, source in enumerate(source_model):
    for target in target_model:
        acc_df.at[i, target] = (model_yp[target][source] == label).astype(np.float32).mean()
        match_df.at[i, target] = (model_yp[target][source] == model_yp[source][source]).astype(np.float32).mean()

acc_df.to_csv('adv_transfer_accuracy.csv', index=False)
match_df.to_csv('adv_transfer_target_matchrate.csv', index=False)