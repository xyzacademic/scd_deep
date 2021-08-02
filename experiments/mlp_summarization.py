import pandas as pd
import os
import matplotlib.pyplot as plt


datasets = ['cifar10_binary', 'stl10_binary']


for dataset in datasets:
    log_path = f'logs/{dataset}'
    title = ['step_size', 'iters', 'batch_size', 'intervals', 'pool_size']
    params = [
        (0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
        (500, 1000, 2000, 4000, 8000),
        (0.05, 0.1, 0.25, 0.5, 0.75, 0.9),
        (5, 10, 20),
        (64, 128, 256),
    ]
    keys = ['lr', 'it', 'nr', 'interval', 'pool']

    for param in zip(title, params, keys):
        train_acc = {}
        test_acc = {}
        train_loss = {}
        test_loss = {}
        for key in param[1]:
            file_name = f'{dataset}_01_{param[2]}{key}_mlp01scd_0.csv'
            df = pd.read_csv(os.path.join(log_path, file_name))
            col_name = f'{key}'
            train_acc[col_name] = df['train acc']
            test_acc[col_name] = df['test acc']
            train_loss[col_name] = df['train loss']
            test_loss[col_name] = df['test loss']

        pd.DataFrame(train_acc).to_csv(os.path.join('results', f'{dataset}_mlp_{param[0]}_train_acc.csv'), index=False)
        pd.DataFrame(test_acc).to_csv(os.path.join('results', f'{dataset}_mlp_{param[0]}_test_acc.csv'), index=False)
        pd.DataFrame(train_loss).to_csv(os.path.join('results', f'{dataset}_mlp_{param[0]}_train_loss.csv'), index=False)
        pd.DataFrame(test_loss).to_csv(os.path.join('results', f'{dataset}_mlp_{param[0]}_test_loss.csv'), index=False)