import sys
sys.path.append('..')
import os
from tools import args, load_data, ModelArgs, BalancedBatchSampler, print_title
import pickle
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch
import numpy as np
from core.ensemble_model import *
import time
from core.text_wrap import TextWrapper

if __name__ == '__main__':
    path = 'checkpoints/pt'
    # args.votes = 2
    # args.target = 'cifar10_toy2_relu_i1_mce_nb1_nw0_dm1_s2_fp32_32'
    # args.version = 'toy2'
    # args.scale = 1
    # args.dataset = 'cifar10'
    # args.n_classes = 10
    # args.act = 'relu'
    # args.cnn = 1
    checkpoints = [os.path.join(path, f'{args.target}_%d.pt' % i) for i in range(args.votes)]
    state_dict = [torch.load(checkpoints[i], map_location=torch.device('cpu')) for i in range(len(checkpoints))]

    structure = arch[args.version]
    params = {
        'num_classes': args.n_classes,
        'votes': args.votes,
        'act': 'sign',
        'sigmoid': False,
        'softmax': True,
    }

    scd = structure(**params)
    layers = scd.layers

    for layer in layers:
        if 'conv' in layer:
            weights = torch.cat([state_dict[i][f'{layer}.weight'] for i in range(args.votes)])
            bias = torch.cat([state_dict[i][f'{layer}.bias'] for i in range(args.votes)])
            scd._modules[layer].weight = torch.nn.Parameter(weights, requires_grad=False)
            scd._modules[layer].bias = torch.nn.Parameter(bias, requires_grad=False)
        elif 'fc' in layer:
            weights = torch.cat([state_dict[i][f'{layer}.weight'] for i in range(args.votes)]).unsqueeze(dim=1)
            bias = torch.cat([state_dict[i][f'{layer}.bias'] for i in range(args.votes)])
            scd._modules[layer].weight = torch.nn.Parameter(weights, requires_grad=False)
            scd._modules[layer].bias = torch.nn.Parameter(bias, requires_grad=False)

    with open(f'checkpoints/{args.target}_0.pkl', 'rb') as f:
        temp_model = pickle.load(f)

    scd = TextWrapper(embedding_layer=temp_model.emb_layer, model=scd,
                      stoi=temp_model.word2id)

    with open(f'checkpoints/{args.target}_{args.votes}.pkl', 'wb') as f:
        pickle.dump(scd, f)
        print(f'{args.target}_{args.votes}.pkl saved successfully')

