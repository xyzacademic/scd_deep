import torch
import os
import time
import numpy as np
import sys
import torch.backends.cudnn as cudnn
from sklearn.metrics import balanced_accuracy_score, accuracy_score
sys.path.append('..')

from tools import args, load_data, ModelArgs, BalancedBatchSampler
from core.lossfunction import ZeroOneLoss, BCELoss, CrossEntropyLoss
from core.cnn01 import LeNet_cifar, _01_init, Toy, Toy2, Toy3, Toy2NP, ToyNP, FConv, FC
from core.train_cnn01_basic_01 import train_single_cnn01
import pickle
import pandas as pd
# Args assignment
scd_args = ModelArgs()

scd_args.nrows = args.nrows
scd_args.local_iter = args.localit
scd_args.num_iters = args.iters
scd_args.interval = args.interval
scd_args.rounds = 1
scd_args.w_inc1 = args.w_inc1
scd_args.updated_fc_features = args.updated_fc_features
scd_args.updated_conv_features = args.updated_conv_features
scd_args.n_jobs = 1
scd_args.num_gpus = 1
scd_args.adv_train = False
scd_args.eps = 0.1
scd_args.w_inc2 = args.w_inc2
scd_args.hidden_nodes = 20
scd_args.evaluation = True
scd_args.verbose = True
scd_args.b_ratio = 0.2
scd_args.cuda = True if torch.cuda.is_available() else False
scd_args.seed = args.seed
scd_args.source = None
scd_args.save = True
scd_args.resume = False
scd_args.criterion = BCELoss if args.loss == 'bce' else ZeroOneLoss
if args.version == 'toy':
    scd_args.structure = Toy
elif args.version == 'toy2':
    scd_args.structure = Toy2
elif args.version == 'toy3':
    scd_args.structure = Toy3
elif args.version == 'lenet':
    scd_args.structure = LeNet_cifar
elif args.version == 'toy2np':
    scd_args.structure = Toy2NP
elif args.version == 'toynp':
    scd_args.structure = ToyNP
elif args.version == 'fc':
    scd_args.structure = FC
scd_args.dataset = args.dataset
scd_args.num_classes = args.n_classes
scd_args.gpu = 0
scd_args.fp16 = True if args.fp16 else False
scd_args.act = args.act
scd_args.updated_fc_nodes = args.updated_fc_nodes
scd_args.updated_conv_nodes = args.updated_conv_nodes
scd_args.width = args.width
scd_args.updated_fc_ratio = 1
scd_args.updated_conv_ratio = 1
scd_args.normal_noise = True
scd_args.verbose = True
scd_args.normalize = bool(args.normalize)
scd_args.batch_size = 256
scd_args.sigmoid = True if args.loss == 'bce' else False
scd_args.softmax = False
scd_args.percentile = bool(args.percentile)
scd_args.fail_count = args.fail_count
scd_args.loss = args.loss
scd_args.diversity = False
scd_args.fc_diversity = bool(args.fc_diversity)
scd_args.conv_diversity = False
scd_args.updated_conv_features_diversity = 16
scd_args.diversity_train_stop_iters = 3000
scd_args.init = args.init
scd_args.target = args.target
scd_args.logs = {}
scd_args.no_bias = args.no_bias
scd_args.record = False
scd_args.scale = args.scale
scd_args.save_path = os.path.join('checkpoints', 'pt')
scd_args.divmean = args.divmean
scd_args.verbose_iter = 20

np.random.seed(scd_args.seed)

train_data, test_data, train_label, test_label = load_data(args.dataset, args.n_classes)

normal_noise = True if args.normal_noise else False
if normal_noise:
    print(f'Normal noise augmented with eps: {args.epsilon}')
    noise = np.random.normal(0, 1, size=train_data.shape)
    noisy = np.clip((train_data + noise * args.epsilon), 0, 1)
    train_data = np.concatenate([train_data, noisy], axis=0)
    train_label = np.concatenate([train_label] * 2, axis=0)
    print(f'train data shape: {train_data.shape}')

if args.cnn:
    train_data = train_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
    test_data = test_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))

if args.divmean:
    train_data = train_data / 0.5 - 1
    test_data = test_data / 0.5 - 1



best_model, val_acc = train_single_cnn01(scd_args, None, None, (train_data, test_data, train_label, test_label))