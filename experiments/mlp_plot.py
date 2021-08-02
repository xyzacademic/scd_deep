import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

save_path = 'figures'

save_path = 'figures'
if not os.path.exists(save_path):
    os.makedirs(save_path)
log_path = 'results'
datasets = [
    'cifar10_binary',
    # 'stl10_binary'
]

# # lr
# keys = ['train_loss', 'test_loss', 'train_acc', 'test_acc']
#
# for dataset in datasets[:1]:
#     for key in keys[:1]:
#         file_name = f'{dataset}_mlp_step_size_{key}.csv'
#         df = pd.read_csv(os.path.join(log_path, file_name))

# # batch size
keys = ['train_loss', 'test_loss', 'train_acc', 'test_acc']
prefix = 'batch_size'

for dataset in datasets:
    figure = plt.figure(figsize=(8, 6), dpi=200)
    i = 1
    for key in keys:
        file_name = f'{dataset}_mlp_{prefix}_{key}.csv'
        df = pd.read_csv(os.path.join(log_path, file_name)).iloc[np.arange(0, 1000, 10).tolist()]
        plt.subplot(2, 2, i)
        df.plot(ax=plt.gca())
        plt.xlabel('iterations')
        plt.ylabel(key.replace('_', ' '))
        # plt.title('Sampling ratio in each training iterations')
        plt.title(dataset.replace('_binary', ' (0 vs 1)'))
        i += 1
    figure_name = f'{dataset}_mlp_{prefix}'
    plt.savefig(os.path.join(save_path, figure_name))
    plt.show()
    plt.close(figure)


# # interval
keys = ['train_loss', 'test_loss', 'train_acc', 'test_acc']
prefix = 'intervals'

for dataset in datasets:
    figure = plt.figure(figsize=(8, 6), dpi=200)
    i = 1
    for key in keys:
        file_name = f'{dataset}_mlp_{prefix}_{key}.csv'
        df = pd.read_csv(os.path.join(log_path, file_name)).iloc[np.arange(0, 1000, 10).tolist()]
        plt.subplot(2, 2, i)
        df.plot(ax=plt.gca())
        plt.xlabel('iterations')
        plt.ylabel(key.replace('_', ' '))
        # plt.title('Sampling ratio in each training iterations')
        plt.title(dataset.replace('_binary', ' (0 vs 1)'))
        i += 1
    figure_name = f'{dataset}_mlp_{prefix}'
    plt.savefig(os.path.join(save_path, figure_name))
    plt.show()
    plt.close(figure)


# # pool size
keys = ['train_loss', 'test_loss', 'train_acc', 'test_acc']
prefix = 'pool_size'

for dataset in datasets:
    figure = plt.figure(figsize=(8, 6), dpi=200)
    i = 1
    for key in keys:
        file_name = f'{dataset}_mlp_{prefix}_{key}.csv'
        df = pd.read_csv(os.path.join(log_path, file_name)).iloc[np.arange(0, 1000, 20).tolist()]
        plt.subplot(2, 2, i)
        df.plot(ax=plt.gca())
        plt.xlabel('iterations')
        plt.ylabel(key.replace('_', ' '))
        # plt.title('Sampling ratio in each training iterations')
        plt.title(dataset.replace('_binary', ' (0 vs 1)'))
        i += 1
    figure_name = f'{dataset}_mlp_{prefix}'
    plt.savefig(os.path.join(save_path, figure_name))
    plt.show()
    plt.close(figure)


# lr
keys = ['train_loss', 'test_loss', 'train_acc', 'test_acc']
prefix = 'step_size'

for dataset in datasets:
    figure = plt.figure(figsize=(8, 6), dpi=200)
    i = 1
    for key in keys:
        file_name = f'{dataset}_mlp_{prefix}_{key}.csv'
        df = pd.read_csv(os.path.join(log_path, file_name)).iloc[np.arange(0, 1000, 20).tolist()]
        plt.subplot(2, 2, i)
        df.plot(ax=plt.gca())
        plt.xlabel('iterations')
        plt.ylabel(key.replace('_', ' '))
        # plt.title('Sampling ratio in each training iterations')
        plt.title(dataset.replace('_binary', ' (0 vs 1)'))
        i += 1
    figure_name = f'{dataset}_mlp_{prefix}'
    plt.savefig(os.path.join(save_path, figure_name))
    plt.show()
    plt.close(figure)


# # iters
keys = ['train_loss', 'test_loss', 'train_acc', 'test_acc']
prefix = 'iters'

for dataset in datasets:
    figure = plt.figure(figsize=(8, 6), dpi=200)
    i = 1
    for key in keys:
        file_name = f'{dataset}_mlp_{prefix}_{key}.csv'
        df = pd.read_csv(os.path.join(log_path, file_name))['8000'].iloc[np.arange(0, 8000, 160).tolist()]
        plt.subplot(2, 2, i)
        df.plot(ax=plt.gca())
        plt.xlabel('iterations')
        plt.ylabel(key.replace('_', ' '))
        # plt.title('Sampling ratio in each training iterations')
        plt.title(dataset.replace('_binary', ' (0 vs 1)'))
        i += 1
    figure_name = f'{dataset}_mlp_{prefix}'
    plt.savefig(os.path.join(save_path, figure_name))
    plt.show()
    plt.close(figure)