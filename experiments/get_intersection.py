import numpy as np
import pickle
import sys
sys.path.append('..')
from tools import args, save_checkpoint, print_title, load_data
import time
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
# from core.cnn_ensemble import CNNVote
# from core.bnn import BNN
import os



if __name__ == '__main__':

    save_path = 'intersection_index'
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except:
            pass

    models = [
        f'{args.dataset}_{args.c0}{args.c1}_mlpbcebp_8',
        f'{args.dataset}_{args.c0}{args.c1}_mlpbceban_8',
        f'{args.dataset}_{args.c0}{args.c1}_mlp01scd_8',
        f'{args.dataset}_{args.c0}{args.c1}_mlpbcescd_8',
    ] if args.version == 'mlp' else [
        f'{args.dataset}_{args.c0}{args.c1}_toy3rrr100_8',
        f'{args.dataset}_{args.c0}{args.c1}_toy3sss100scale_ban_8',
        f'{args.dataset}_{args.c0}{args.c1}_toy3sss100scale_abp_8',
    ]
    np.random.seed(2019)

    train_data, test_data, train_label, test_label = load_data(args.dataset, args.n_classes, c1=args.c0, c2=args.c1)
    if args.cnn:
        if 'cifar10' in args.dataset:
            train_data = train_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
            test_data = test_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
        elif 'stl10' in args.dataset:
            train_data = train_data.reshape((-1, 96, 96, 3)).transpose((0, 3, 1, 2))
            test_data = test_data.reshape((-1, 96, 96, 3)).transpose((0, 3, 1, 2))



    correct_index = []

    for model in models:
        print(model)


        with open('checkpoints/%s.pkl' % model, 'rb') as f:
            scd = pickle.load(f)
            # scd.best_model.status = 'sign'
            # for i in range(len(scd.models)):
            #     scd.models[i].status = 'sign'
        if 'dm1' in model:
            yp = scd.predict(test_data / 0.5 - 1)
        else:
            yp = scd.predict(test_data)
        print(f'Model: {model}')
        print('test accuracy: ', accuracy_score(test_label, yp))
        del scd
        correct_index.append((yp == test_label).astype(np.int8))

    correct_index = np.stack(correct_index, axis=1).sum(axis=1) // len(models)
    correct_index = np.nonzero(correct_index)[0]
    np.save(os.path.join(save_path, f'{args.dataset}_{args.version}_{args.c0}{args.c1}_correct.npy'), correct_index)

    pred_one_hot = np.eye(args.n_classes)[test_label[correct_index]].astype(np.int8)

    index_in_class = []
    for i in range(args.n_classes):
        index_in_class.append(correct_index[pred_one_hot[:, i].astype(np.bool)])

    reserve = np.stack([np.random.choice(index_in_class[i], 50, False) for i in range(args.n_classes)], axis=1)
    reserve = reserve.flatten()
    np.save(os.path.join(save_path, f'{args.dataset}_{args.version}_{args.c0}{args.c1}.npy'), reserve)
    # re_f = reserve.astype(np.str).tolist()
    # idx = ", ".join(re_f)
    #
    # with open('hsj_example_mc_idx', 'w') as f:
    #     f.write(idx + '\n')



