import pandas as pd
import numpy as np
import os



"""
white box attack
"""

# MLP

# dataset = 'stl10_binary'
# log_path = f'../PyTorch_CIFAR10/results/outer/{dataset}'
# method = 'fgsm'
# version = 'mlp'
# models = ['mlpbcebp', 'mlpbceban', 'mlp01scd', 'mlpbcescd']
# class_pairs = []
# save_path = 'results'
#
#
# for c0 in range(9):
#     for c1 in range(c0+1, 10):
#         class_pairs.append((c0, c1))
#
# df = pd.DataFrame(columns=['classes'] + models)
# df['classes'] = [f'{c0} vs {c1}' for (c0, c1) in class_pairs]
# for i, class_pair in enumerate(class_pairs):
#     temp_df = pd.read_csv(os.path.join(log_path, f'{version}_{method}_16_20_{class_pair[0]}{class_pair[1]}.csv'))
#     for j, model in enumerate(models):
#         df.at[i, model] = temp_df.at[j, model] * 100
#
# df.to_csv(os.path.join(save_path, f'{dataset}_{version}_{method}_whitebox.csv'), index=False)

# CNN

# dataset = 'cifar10_binary'
# log_path = f'../PyTorch_CIFAR10/results/outer/{dataset}'
# method = 'fgsm'
# version = 'cnn'
# models = ['toy3rrr100', 'toy3sss100scale_ban', 'toy3sss100scale_abp']
# class_pairs = []
# save_path = 'results'
#
#
# for c0 in range(9):
#     for c1 in range(c0+1, 10):
#         class_pairs.append((c0, c1))
#
# df = pd.DataFrame(columns=['classes'] + models)
# df['classes'] = [f'{c0} vs {c1}' for (c0, c1) in class_pairs]
# for i, class_pair in enumerate(class_pairs):
#     temp_df = pd.read_csv(os.path.join(log_path, f'{version}_{method}_16_20_{class_pair[0]}{class_pair[1]}.csv'))
#     for j, model in enumerate(models):
#         df.at[i, model] = temp_df.at[j, model] * 100
# df.columns = ['classes', 'cnnbcebp', 'cnnbceban', 'cnn01abp']
# df.to_csv(os.path.join(save_path, f'{dataset}_{version}_{method}_whitebox.csv'), index=False)


"""
Transfer attack
"""

# MLP
# dataset = 'cifar10_binary'
# log_path = f'../PyTorch_CIFAR10/results/outer/{dataset}'
# method = 'fgsm'
# version = 'mlp'
# models = ['mlpbcebp', 'mlpbceban', 'mlp01scd', 'mlpbcescd']
# class_pairs = []
# save_path = 'results'
#
#
# for c0 in range(9):
#     for c1 in range(c0+1, 10):
#         class_pairs.append((c0, c1))
#
# for j, target_model in enumerate(models):
#
#     df = pd.DataFrame(columns=['classes\\target'] + models)
#     df['classes\\target'] = [f'{c0} vs {c1}' for (c0, c1) in class_pairs]
#     for i, class_pair in enumerate(class_pairs):
#         temp_df = pd.read_csv(os.path.join(log_path, f'{version}_{method}_16_20_{class_pair[0]}{class_pair[1]}.csv'))
#         for j, model in enumerate(models):
#             df.at[i, model] = temp_df.at[j, target_model] * 100
#     df = df.drop(columns=[target_model], axis=1)
#     df.to_csv(os.path.join(save_path, f'{dataset}_{version}_{method}_{target_model}_transfer.csv'), index=False)

# CNN

# dataset = 'cifar10_binary'
# log_path = f'../PyTorch_CIFAR10/results/outer/{dataset}'
# method = 'pgd'
# version = 'cnn'
# models = ['toy3rrr100', 'toy3sss100scale_ban', 'toy3sss100scale_abp']
# class_pairs = []
# save_path = 'results'
# 
# 
# for c0 in range(9):
#     for c1 in range(c0+1, 10):
#         class_pairs.append((c0, c1))
# 
# for j, target_model in enumerate(models):
# 
#     df = pd.DataFrame(columns=['classes\\target'] + models)
#     df['classes\\target'] = [f'{c0} vs {c1}' for (c0, c1) in class_pairs]
#     for i, class_pair in enumerate(class_pairs):
#         temp_df = pd.read_csv(os.path.join(log_path, f'{version}_{method}_16_20_{class_pair[0]}{class_pair[1]}.csv'))
#         for j, model in enumerate(models):
#             df.at[i, model] = temp_df.at[j, target_model] * 100
#     df = df.drop(columns=[target_model], axis=1)
#     df.to_csv(os.path.join(save_path, f'{dataset}_{version}_{method}_{target_model}_transfer.csv'), index=False)


# """
# inner transfer
# """
#
# # MLP
#
# dataset = 'stl10_binary'
# log_path = f'../PyTorch_CIFAR10/results/inner/{dataset}'
# method = 'pgd'
# version = 'mlp'
# models = ['mlpbcebp', 'mlpbceban', 'mlp01scd', 'mlpbcescd']
# class_pairs = []
# save_path = 'results'
#
# mask = 1-np.eye(8)
# for c0 in range(9):
#     for c1 in range(c0+1, 10):
#         class_pairs.append((c0, c1))
#
#     df = pd.DataFrame(columns=['classes'] + models)
#     df['classes'] = [f'{c0} vs {c1}' for (c0, c1) in class_pairs]
#
#     for i, class_pair in enumerate(class_pairs):
#
#         for j, model in enumerate(models):
#             temp_df = pd.read_csv(
#                 os.path.join(log_path, f'{model}_{method}_16_20_{class_pair[0]}{class_pair[1]}.csv'))
#             df.at[i, model] = (temp_df.values[:, 1:] * mask).sum() / mask.sum() * 100
#
# df.to_csv(os.path.join(save_path, f'{dataset}_{version}_{method}_inner_transfer.csv'), index=False)


#MLP

# dataset = 'cifar10_binary'
# log_path = f'../PyTorch_CIFAR10/results/inner/{dataset}'
# method = 'fgsm'
# version = 'cnn'
# models = ['toy3rrr100', 'toy3sss100scale_ban', 'toy3sss100scale_abp']
# class_pairs = []
# save_path = 'results'
#
# mask = 1-np.eye(8)
# for c0 in range(9):
#     for c1 in range(c0+1, 10):
#         class_pairs.append((c0, c1))
#
#     df = pd.DataFrame(columns=['classes'] + models)
#     df['classes'] = [f'{c0} vs {c1}' for (c0, c1) in class_pairs]
#
#     for i, class_pair in enumerate(class_pairs):
#
#         for j, model in enumerate(models):
#             temp_df = pd.read_csv(
#                 os.path.join(log_path, f'{model}_{method}_16_20_{class_pair[0]}{class_pair[1]}.csv'))
#             df.at[i, model] = (temp_df.values[:, 1:] * mask).sum() / mask.sum() * 100
# df.columns = ['classes', 'cnnbcebp', 'cnnbceban', 'cnn01abp']
# df.to_csv(os.path.join(save_path, f'{dataset}_{version}_{method}_inner_transfer.csv'), index=False)


"""
outer transfer
"""

# # MLP
# 
dataset = 'cifar10_binary'
log_path = f'../PyTorch_CIFAR10/results/outer/{dataset}'
method = 'pgd'
version = 'mlp'
models = ['mlpbcebp', 'mlpbceban', 'mlp01scd', 'mlpbcescd']
class_pairs = []
save_path = 'results'


for c0 in range(9):
    for c1 in range(c0+1, 10):
        class_pairs.append((c0, c1))

df = pd.DataFrame(columns=['target'] +
                          ['mlpbcebp-mlpbceban', 'mlpbcebp-mlp01scd', 'mlpbcebp-mlpbcescd'] +
                  ['mlpbceban-mlpbcebp', 'mlpbceban-mlp01scd', 'mlpbceban-mlpbcescd'] +
                          ['mlp01scd-mlpbcebp', 'mlp01scd-mlpbceban', 'mlp01scd-mlpbcescd'] +
                  ['mlpbcescd-mlpbcebp', 'mlpbcescd-mlpbceban', 'mlpbcescd-mlp01scd'])


df['target'] = [f'{c0} vs {c1}' for (c0, c1) in class_pairs]

for i, class_pair in enumerate(class_pairs):


        temp_df = pd.read_csv(
            os.path.join(log_path, f'{version}_{method}_16_20_{class_pair[0]}{class_pair[1]}.csv'))
        for j, model in enumerate(models):
            for model_a in models:
                if f'{model}-{model_a}' in df.columns:
                    df.at[i, f'{model}-{model_a}'] = temp_df.at[j, model_a] * 100

df.columns = ['target'] + ['mlpbceban', 'mlp01scd', 'mlpbcescd'] + \
                  ['mlpbcebp', 'mlp01scd', 'mlpbcescd'] + \
                          ['mlpbcebp', 'mlpbceban', 'mlpbcescd'] + \
                  ['mlpbcebp', 'mlpbceban', 'mlp01scd']
df.to_csv(os.path.join(save_path, f'{dataset}_{version}_{method}_outer_transfer.csv'), index=False)


# cnn

# dataset = 'cifar10_binary'
# log_path = f'../PyTorch_CIFAR10/results/outer/{dataset}'
# method = 'fgsm'
# version = 'cnn'
# models = ['toy3rrr100', 'toy3sss100scale_ban', 'toy3sss100scale_abp']
# class_pairs = []
# save_path = 'results'
#
#
# for c0 in range(9):
#     for c1 in range(c0+1, 10):
#         class_pairs.append((c0, c1))
#
# df = pd.DataFrame(columns=['target'] +
#                           ['toy3rrr100-toy3sss100scale_ban', 'toy3rrr100-toy3sss100scale_abp'] +
#                   ['toy3sss100scale_ban-toy3rrr100', 'toy3sss100scale_ban-toy3sss100scale_abp'] +
#                           ['toy3sss100scale_abp-toy3rrr100', 'toy3sss100scale_abp-toy3sss100scale_ban'] )
#
#
# df['target'] = [f'{c0} vs {c1}' for (c0, c1) in class_pairs]
#
# for i, class_pair in enumerate(class_pairs):
#
#
#         temp_df = pd.read_csv(
#             os.path.join(log_path, f'{version}_{method}_16_20_{class_pair[0]}{class_pair[1]}.csv'))
#         for j, model in enumerate(models):
#             for model_a in models:
#                 if f'{model}-{model_a}' in df.columns:
#                     df.at[i, f'{model}-{model_a}'] = temp_df.at[j, model_a] * 100
#
# df.columns = ['target'] + ['cnnbceban', 'cnn01abp'] + \
#              ['cnnbcebp', 'cnn01abp'] + \
#              ['cnnbcebp', 'cnnbceban']
# df.to_csv(os.path.join(save_path, f'{dataset}_{version}_{method}_outer_transfer.csv'), index=False)