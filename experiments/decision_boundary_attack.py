import sys
sys.path.append('..')
import pickle
import os
from tools import args, save_checkpoint, print_title, load_data
import numpy as np
from art.attacks.evasion import BoundaryAttack
from art.classifiers import BlackBoxClassifier
import time
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

class modelWrapper():
    def __init__(self, model, dm1=False):
        self.model = model
        self.dm1 = dm1

    def predict_one_hot(self, x_test, **kwargs):
        if self.dm1:
            pred_y = self.model.predict(x_test / 0.5 - 1, **kwargs)
        else:
            pred_y = self.model.predict(x_test, **kwargs)
        pred_one_hot = np.eye(args.n_classes)[pred_y.astype(int)]

        return pred_one_hot


if __name__ == '__main__':

    # load data
    train, test, train_label, test_label = load_data(args.dataset, args.n_classes, c1=args.c0, c2=args.c1)
    if args.cnn:
        if 'cifar10' in args.dataset:
            train = train.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
            test = test.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
        elif 'stl10' in args.dataset:
            train = train.reshape((-1, 96, 96, 3)).transpose((0, 3, 1, 2))
            test = test.reshape((-1, 96, 96, 3)).transpose((0, 3, 1, 2))
    with open(f'checkpoints/{args.target}', 'rb') as f:
        scd = pickle.load(f)

    print(f'Load model {args.target}')
    # get vector shape
    input_shape = train.shape[1:]
    np.random.seed(args.seed)
    # print accuracy
    if 'dm1' in args.target:
        yp = scd.predict(test / 0.5 - 1)
    else:
        yp = scd.predict(test)
    acc = accuracy_score(y_true=test_label, y_pred=yp)
    print(f'{args.votes} votes test accuracy: {acc}')
    # correct_index = np.nonzero((yp == test_label).astype(np.int8))[0]
    correct_index = np.load(f'intersection_index/{args.dataset}_{args.version}_{args.c0}{args.c1}_correct.npy')

    init_adv_c = [None for i in range(args.n_classes)]
    if args.adv_init:
        found = 0
        for index in correct_index:
            for i in range(args.n_classes):
                if test_label[index] == i and init_adv_c[i] is None:
                    init_adv_c[i] = test[index]
                    found += 1
                    break
            if found >= args.n_classes:
                break
        init_adv_c = np.stack(init_adv_c, axis=0)
        if 'dm1' in args.target:
            yp = scd.predict(init_adv_c / 0.5 - 1)
        else:
            yp = scd.predict(init_adv_c)

        for i in range(args.n_classes):
            print(f'init_adv_c{i} prediction: ', yp[i])


    predictWrapper = modelWrapper(scd, 'dm1' in args.target)

    # get vector value range
    min_pixel_value = train.min()
    max_pixel_value = train.max()
    print('min_pixel_value ', min_pixel_value)
    print('max_pixel_value ', max_pixel_value)

    log_path = args.source
    if not os.path.exists(log_path):
        try:
            os.makedirs(log_path)
        except:
            pass

    # Create classifier
    classifier = BlackBoxClassifier(predict=predictWrapper.predict_one_hot,
                                    input_shape=input_shape,
                                    nb_classes=args.n_classes,
                                    # channels_first=True,
                                    clip_values=(min_pixel_value, max_pixel_value))

    print('----- generate adv data by HopSkipJump attack -----')
    # Generate adversarial test examples
    # N runs

    l2_ = []
    linf_ = []

    for i in range(args.round):
        print(f"Round #{i}")
        # attacker = HopSkipJump(classifier=classifier, targeted=False, norm=2, max_iter=args.iters,
        #                        max_eval=10000, init_eval=100,
        #                        init_size=args.train_size)
        attacker = BoundaryAttack(estimator=classifier, targeted=False, max_iter=args.iters,
                                  delta=0.01, epsilon=0.01, step_adapt=0.2, num_trial=100,
                                  sample_size=100, init_size=args.train_size, )
        # attacker = HopSkipJump(classifier=classifier, targeted=False, norm=2, max_iter=2, max_eval=10000, init_eval=100, init_size=100)

        # Input data shape should be 2D

        datapoint = test[args.index:args.index+1]

        s = time.time()
        if args.adv_init:
            print('Given x_adv_init')
            ci = test_label[args.index]
            adv_data = attacker.generate(x=datapoint, x_adv_init=np.expand_dims(init_adv_c[ci-1], 0))
        else:
            print('Random init')
            adv_data = attacker.generate(x=datapoint)

        # distortion(datapoint, adv_data)
        print('Generate test adv cost time: ', time.time() - s)
        print(f'Index: {args.index}')
        print(f'True label: {test_label[args.index]}')
        if 'dm1' in args.target:
            pc = scd.predict(datapoint / 0.5 - 1)
            pa = scd.predict(adv_data / 0.5 - 1)
        else:
            pc = scd.predict(datapoint)
            pa = scd.predict(adv_data)
        print(f'Prediction on clean: {pc}')
        print(f'Prediction on adv: {pa}')
        diff = (datapoint - adv_data).reshape((datapoint.shape[0], -1))
        l2_distance = np.linalg.norm(diff, axis=1)[0]
        linf_distance = np.max(np.abs(diff), axis=1)[0]
        if pc != pa:
            print('L2: ', l2_distance)
            print('Linf: ', linf_distance)
            l2_.append(l2_distance)
            linf_.append(linf_distance)
        else:
            print('Attacking failed')
            l2_.append('failed')
            linf_.append('failed')

        # return adv_data
        adv_path = f'db_adv_data/{args.iters}/{args.target.replace(".pkl", "")}'
        # return adv_data
        if not os.path.exists(adv_path):
            try:
                os.makedirs(adv_path)
            except:
                pass
        np.save(os.path.join(adv_path, f'{args.index}.npy'), adv_data)
    # save logs

    file_name = args.target.replace('.pkl', '_id#%s.csv' % args.index)
    if os.path.exists(os.path.join(log_path, file_name)):
        df = pd.read_csv(os.path.join(log_path, file_name))
        df['%s votes' % args.votes] = ['L2 distance'] + l2_
        df['%.4f' % (acc)] = ['Linf distance:'] + linf_
    else:
        df = pd.DataFrame(
            {
                'round': np.arange(args.round+1),
                '%s votes' % args.votes: ['L2 distance'] + l2_,
                '%.4f' % (acc): ['Linf distance:'] + linf_,

            })

    df.to_csv(os.path.join(log_path, file_name), index=False)