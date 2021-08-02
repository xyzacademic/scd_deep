import os
import numpy as np
import pandas as pd

idx_path = 'intersection_index'

class_pairs = []
for c0 in range(9):
    for c1 in range(c0+1, 10):
        class_pairs.append((c0, c1))

dataset = 'cifar10_binary'
version = 'mlp'
attack = 'db'
log_path = f'{attack}_fi_attack_logs_100'
models = [
    f'mlpbcebp',
    f'mlpbceban',
    f'mlp01scd',
    f'mlpbcescd',
] if version == 'mlp' else [
    f'toy3rrr100',
    f'toy3sss100scale_ban',
    f'toy3sss100scale_abp',
]

df = pd.DataFrame(columns=['class'] + models)
df['class'] = [f'{class_pair[0]} vs {class_pair[1]}' for class_pair in class_pairs]

for i, class_pair in enumerate(class_pairs):
    # indices = np.load(os.path.join(idx_path, f'{dataset}_{version}_{class_pair[0]}{class_pair[1]}.npy'))
    values = {}
    for model in models:
        values[model] = []


    for model in models:
        files = [file for file in os.listdir(log_path) if f'{dataset}_{class_pair[0]}{class_pair[1]}_{model}' in file]
        for file in files:
            try:
                temp_df = pd.read_csv(os.path.join(log_path, file))
                values[model].append(float(temp_df.iloc[1][1]))
            except:
                pass

    for model in models:
        print(f'class #{i}, model: {model}, #records: {len(values[model])}')
        # values[model] = sum(values[model]) / (len(values[model])+1)
        # print(values[model])
        values[model] = np.nanmedian(values[model])
        df.at[i, model] = values[model]
        
df.to_csv(f'results/{dataset}_{version}_{attack}.csv', index=False)