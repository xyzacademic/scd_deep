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

datasets = ['cifar10_binary', 'stl10_binary']
class_pairs = []
for c0 in range(9):
    for c1 in range(c0+1, 10):
        class_pairs.append((c0, c1))


for dataset in datasets:
    train_df = pd.DataFrame(columns=['classes'] + ['mlpbcebp', 'mlpbceban', 'mlp01scd', 'mlpbcescd'])
    train_df['classes'] = [f'{c0} vs {c1}' for (c0, c1) in class_pairs]
    test_df = pd.DataFrame(columns=['classes'] + ['mlpbcebp', 'mlpbceban', 'mlp01scd', 'mlpbcescd'])
    test_df['classes'] = [f'{c0} vs {c1}' for (c0, c1) in class_pairs]
    for (c0, c1) in class_pairs:
        for model_name in ['mlpbcebp', 'mlpbceban', 'mlp01scd', 'mlpbcescd']:
            with open(f'checkpoints/{dataset}_{c0}{c1}_{model_name}_8.pkl', 'rb') as f:
                model = pickle.load(f)
            train_data, test_data, train_label, test_label = load_data(dataset, 2, c1=c0, c2=c1)

            yp = model.predict(train_data, batch_size=200)

            train_acc = accuracy_score(y_true=train_label, y_pred=yp)
            train_df.loc[train_df['classes'] == f'{c0} vs {c1}', model_name] = train_acc

            yp = model.predict(test_data, batch_size=200)

            test_acc = accuracy_score(y_true=test_label, y_pred=yp)
            test_df.loc[train_df['classes'] == f'{c0} vs {c1}', model_name] = test_acc

            del model

    train_df.to_csv(os.path.join(save_path, f'{dataset}_mlp_train.csv'), index=False)
    test_df.to_csv(os.path.join(save_path, f'{dataset}_mlp_test.csv'), index=False)


for dataset in datasets:
    train_df = pd.read_csv(os.path.join(save_path, f'{dataset}_mlp_train.csv'))
    test_df = pd.read_csv(os.path.join(save_path, f'{dataset}_mlp_test.csv'))
    df = train_df.copy()
    for (c0, c1) in class_pairs:
        for model_name in ['mlpbcebp', 'mlpbceban', 'mlp01scd', 'mlpbcescd']:
            df.loc[(df.classes == f'{c0} vs {c1}'), model_name] = \
                f'%.2f (%.2f)' % (train_df.loc[(train_df.classes == f'{c0} vs {c1}'), model_name]*100,
                                  test_df.loc[(test_df.classes == f'{c0} vs {c1}'), model_name]*100)

    df.to_csv(os.path.join(save_path, f'{dataset}_mlp.csv'), index=False)

d1 = pd.read_csv(os.path.join(save_path, 'cifar10_binary_mlp.csv'))
d2 = pd.read_csv(os.path.join(save_path, 'stl10_binary_mlp.csv'))
df = pd.concat([d1, d2], axis=1)
df.to_csv(os.path.join(save_path, 'mlp.csv'), index=False)

