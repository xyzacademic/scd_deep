import sys
sys.path.append('..')
from core.cnn import Toy3BN
from core.cnn01 import *
import os
from tools import args, load_data, ModelArgs, BalancedBatchSampler, print_title
import pickle
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch
import numpy as np
from core.ensemble_model import ModelWrapper2
import time

if __name__ == '__main__':
    path = 'checkpoints/pt'
    # args.target = 'cifar10_toy3_i3_bce_div_nb_s2_16'
    checkpoints = [os.path.join(path, f'{args.target}_%d.pt' % i) for i in range(args.votes)]
    # names = [name for name in os.listdir(path) if 'toy2' in name]
    # print(len(names))
    # checkpoints = [os.path.join(path, name) for name in names]
    # print(len(checkpoints))

    version = arch[args.version]

    scd = ModelWrapper2(
        structure=version, votes=args.votes, path=checkpoints)

    with open(f'checkpoints/{args.target}_{args.votes}.pkl', 'wb') as f:
        pickle.dump(scd, f)
        print(f'{args.target}_{args.votes}.pkl save successfully')

    train_data, test_data, train_label, test_label = load_data(args.dataset, args.n_classes)
    if args.cnn:
        train_data = train_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
        test_data = test_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))

    if 'dm1' in args.target:
        train_data = train_data / 0.5 - 1
        test_data = test_data / 0.5 - 1

    start_time = time.time()
    yp = scd.predict(train_data, batch_size=1000)
    end_time = time.time()
    train_acc = accuracy_score(y_true=train_label, y_pred=yp)
    print(f'{args.votes} votes train accuracy, '
          f'{train_acc} '
          f'inference cost %.2f seconds: ' % (end_time - start_time),
          )

    start_time = time.time()
    yp = scd.predict(test_data, batch_size=1000)
    end_time = time.time()
    test_acc = accuracy_score(y_true=test_label, y_pred=yp)
    print(f'{args.votes} votes test accuracy: '
          f'{test_acc} '
          f'inference cost %.2f seconds: ' % (end_time - start_time),
          )

    save = True if args.save else False

    if save:
        path = os.path.join('logs', 'combined_acc')
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except:
                pass
        with open(os.path.join(path, f'{args.target}_{args.votes}'), 'w') as f:
            f.write('%.4f %.4f' % (train_acc, test_acc))