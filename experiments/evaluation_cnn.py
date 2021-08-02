import pandas as pd
import pickle
import sys
sys.path.append('..')
from tools import args, load_data
import os
from sklearn.metrics import accuracy_score

save_path = 'results'
if not os.path.exists(save_path):
    os.makedirs(save_path)

datasets = [('cifar10_binary', ['toy3rrr100', 'toy3sss100scale_ban', 'toy3sss100scale_abp']),

            ('stl10_binary', ['toy4rrr100', 'toy4ssss100scale_ban', 'toy4ssss100scale_abp'])]
class_pairs = []
for c0 in range(9):
    for c1 in range(c0+1, 10):
        class_pairs.append((c0, c1))


# for dataset, models in datasets:
#
#     train_df = pd.DataFrame(columns=['classes'] + models)
#     train_df['classes'] = [f'{c0} vs {c1}' for (c0, c1) in class_pairs]
#     test_df = pd.DataFrame(columns=['classes'] + models)
#     test_df['classes'] = [f'{c0} vs {c1}' for (c0, c1) in class_pairs]
#     for (c0, c1) in class_pairs:
#         print(f'class pairs: {c0} vs {c1}')
#         for model_name in models:
#             with open(f'checkpoints/{dataset}_{c0}{c1}_{model_name}_8.pkl', 'rb') as f:
#                 model = pickle.load(f)
#             train_data, test_data, train_label, test_label = load_data(dataset, 2, c1=c0, c2=c1)
#             if 'cifar10' in dataset:
#                 train_data = train_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
#                 test_data = test_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
#             elif 'stl10' in dataset:
#                 train_data = train_data.reshape((-1, 96, 96, 3)).transpose((0, 3, 1, 2))
#                 test_data = test_data.reshape((-1, 96, 96, 3)).transpose((0, 3, 1, 2))
#
#             yp = model.predict(train_data, batch_size=200)
#
#             train_acc = accuracy_score(y_true=train_label, y_pred=yp)
#             train_df.loc[train_df['classes'] == f'{c0} vs {c1}', model_name] = train_acc
#
#             yp = model.predict(test_data, batch_size=200)
#
#             test_acc = accuracy_score(y_true=test_label, y_pred=yp)
#             test_df.loc[train_df['classes'] == f'{c0} vs {c1}', model_name] = test_acc
#
#             del model
#     train_df.columns = ['classes', 'cnnbcebp', 'cnnbceban', 'cnn01abp']
#     test_df.columns = ['classes', 'cnnbcebp', 'cnnbceban', 'cnn01abp']
#     train_df.to_csv(os.path.join(save_path, f'{dataset}_cnn_train.csv'), index=False)
#     test_df.to_csv(os.path.join(save_path, f'{dataset}_cnn_test.csv'), index=False)


for dataset, models in datasets:
    train_df = pd.read_csv(os.path.join(save_path, f'{dataset}_cnn_train.csv'))
    test_df = pd.read_csv(os.path.join(save_path, f'{dataset}_cnn_test.csv'))
    df = train_df.copy()
    for (c0, c1) in class_pairs:
        for model_name in ['cnnbcebp', 'cnnbceban', 'cnn01abp']:
            df.loc[(df.classes == f'{c0} vs {c1}'), model_name] = \
                f'%.2f (%.2f)' % (train_df.loc[(train_df.classes == f'{c0} vs {c1}'), model_name]*100,
                                  test_df.loc[(test_df.classes == f'{c0} vs {c1}'), model_name]*100)

    df.to_csv(os.path.join(save_path, f'{dataset}_cnn.csv'), index=False)

d1 = pd.read_csv(os.path.join(save_path, 'cifar10_binary_cnn.csv'))
d2 = pd.read_csv(os.path.join(save_path, 'stl10_binary_cnn.csv'))
df = pd.concat([d1, d2], axis=1)
df.to_csv(os.path.join(save_path, 'cnn.csv'), index=False)
