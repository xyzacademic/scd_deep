import sys
sys.path.append('..')
from core.cnn01 import LeNet_cifar, Toy, Toy2, Ensemble, Toy3, FConv, FC
import os
from tools import args, load_data, ModelArgs, BalancedBatchSampler, print_title
import pickle
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch
import numpy as np


if __name__ == '__main__':
    path = 'checkpoints/pt'
    # args.target = 'cifar10_toy3_i3_bce_div_nb_s2_16'
    checkpoints = [os.path.join(path, f'{args.target}_%d.pt' % i) for i in range(args.votes)]
    # names = [name for name in os.listdir(path) if 'toy2' in name]
    # print(len(names))
    # checkpoints = [os.path.join(path, name) for name in names]
    # print(len(checkpoints))
    if args.version == 'toy':
        version = Toy
    elif args.version == 'toy2':
        version = Toy2
    elif args.version == 'toy3':
        version = Toy3
    elif args.version == 'fc':
        version = FC
    params = {
        'num_classes': 1 if args.n_classes == 2 else args.n_classes,
        'act': args.act,
        'sigmoid': True if 'bce' in args.target else False,
        'softmax': True if 'mce' in args.target else False,
        'scale': args.scale,
        'divmean': 1 if 'dm1' in args.target else 0,
        'eval': 1
    }

    scd = Ensemble(
        structure=version, params=params, path=checkpoints)

    with open(f'checkpoints/{args.target}.pkl', 'wb') as f:
        pickle.dump(scd, f)
        print(f'{args.target} save successfully')

    train_data, test_data, train_label, test_label = load_data(args.dataset, args.n_classes)
    if args.cnn:
        train_data = train_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
        test_data = test_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))

    if 'dm1' in args.target:
        train_data = train_data / 0.5 - 1
        test_data = test_data / 0.5 - 1

    for i in [1, 3, 16, 32]:
        print(f'{i} votes train accuracy: ', 
              accuracy_score(y_true=train_label, y_pred=scd.predict(train_data, votes=i)))
        print(f'{i} votes test accuracy: ',
              accuracy_score(y_true=test_label, y_pred=scd.predict(test_data, votes=i)))
        print()