import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import time
import numpy as np
import sys
import torch.backends.cudnn as cudnn
from sklearn.metrics import balanced_accuracy_score, accuracy_score

sys.path.append('..')

from tools import args, load_data, ModelArgs, BalancedBatchSampler, print_title, MultiClassSampler, MultiClassSampler2
from core.lossfunction import *
from core.cnn01 import *
from core.basic_module import *
from core.basic_function import evaluation, get_features
import pickle
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from core.src.attack.fast_gradient_sign_untargeted import FastGradientSignUntargeted


def train_single_cnn01(scd_args, device=None, seed=None, data_set=None, fined_train_label=None,
                       target_class=None):
    if device is not None:
        scd_args.gpu = device
    if seed is not None:
        scd_args.seed = seed
    resume = False
    use_cuda = scd_args.cuda
    dtype = torch.float16 if scd_args.fp16 else torch.float32

    best_acc = 0

    # seed = 2047775

    print('Random seed: ', scd_args.seed)
    np.random.seed(scd_args.seed)
    torch.manual_seed(scd_args.seed)
    df = pd.DataFrame(columns=['epoch', 'train acc', 'test acc', 'train loss', 'test loss'])
    log_path = os.path.join('logs', scd_args.dataset)
    log_file_name = os.path.join(log_path, scd_args.target)

    if scd_args.save:
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    train_data, test_data, train_label, test_label = data_set
    batch_size = int(train_data.shape[0] * scd_args.nrows) \
        if scd_args.nrows < 1 else int(scd_args.nrows)


    trainset = TensorDataset(torch.from_numpy(train_data.astype(np.float32)),
                             torch.from_numpy(train_label.astype(np.int64)))
    testset = TensorDataset(torch.from_numpy(test_data.astype(np.float32)),
                            torch.from_numpy(test_label.astype(np.int64)))
    train_transform = transforms.Compose(
        [
            # transforms.ToPILImage(),
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            # transforms.RandomApply([
            #     transforms.Lambda(
            #         lambda x: torch.clamp(x + torch.randn_like(x) * scd_args.eps, min=0, max=1))],
            #     p=0.5)
            transforms.RandomApply([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ],
            p=0.5)
        ])


    train_loader = MultiClassSampler(dataset=trainset, classes=scd_args.num_classes,
                                     nrows=scd_args.nrows, balanced=scd_args.balanced_sampling,
                                     transform=train_transform if scd_args.aug else None)
    val_loader = DataLoader(trainset, batch_size=train_data.shape[0]//scd_args.updated_conv_ratio, shuffle=False, num_workers=0,
                            pin_memory=False)
    test_loader = DataLoader(testset, batch_size=test_data.shape[0]//scd_args.updated_conv_ratio, shuffle=False, num_workers=0,
                             pin_memory=False)

    net = scd_args.structure(
        num_classes=scd_args.num_classes, act=scd_args.act, sigmoid=scd_args.sigmoid,
        softmax=scd_args.softmax, scale=scd_args.scale)
    # if not scd_args.sigmoid and not scd_args.softmax:
    #     _01_init(net)
    # else:
    init_weights(net, kind=scd_args.init)
    best_model = scd_args.structure(
        num_classes=scd_args.num_classes, act=scd_args.act, sigmoid=scd_args.sigmoid,
    softmax=scd_args.softmax, scale=scd_args.scale)

    class Dm(nn.Module):
        def __init__(self):
            super(Dm, self).__init__()

        def forward(self, x):
            return x / 0.5 - 1


    if scd_args.resume:

        temp = torch.load(os.path.join(scd_args.save_path, scd_args.source + '.pt'),
                          map_location=torch.device('cpu'))
        print(f'Load state dict {scd_args.source} successfully')
        net.load_state_dict(temp)

    criterion = scd_args.criterion()

    if scd_args.cuda:
        print('start move to cuda')
        torch.cuda.manual_seed_all(scd_args.seed)
        # torch.backends.cudnn.deterministic = True
        cudnn.benchmark = True
        if scd_args.fp16:
            net = net.half()

        # net = torch.nn.DataParallel(net, device_ids=[0,1])
        device = torch.device("cuda:%s" % scd_args.gpu)
        net.to(device=device)
        best_model.to(device=device)
        criterion.to(device=device, dtype=dtype)

    # normalization
    if scd_args.divmean:
        adv_net = nn.Sequential(Dm(), net)
    else:
        adv_net = net

    net.eval()

    best_acc = 0

    layers = net.layers[::-1]  # reverse the order of layers' name
    if scd_args.bnn_layer > scd_args.freeze_layer:
        print(f'BNN training for {layers[len(layers) - scd_args.bnn_layer]}')

    attacker = FastGradientSignUntargeted(adv_net,
                                        scd_args.epsilon,
                                        scd_args.alpha,
                                        min_val=0,
                                        max_val=1,
                                        max_iters=scd_args.step,
                                        _type='l2')
    # Training
    for epoch in range(scd_args.num_iters):

        # print(f'\nEpoch: {epoch}')
        start_time = time.time()



        if epoch > scd_args.diversity_train_stop_iters:
            scd_args.fc_diversity = False
            scd_args.conv_diversity = False

        if scd_args.lr_decay_iter and (epoch + 1) % scd_args.lr_decay_iter == 0:
            scd_args.w_inc1 /= 2
            scd_args.w_inc2 /= 2
            print(f'w_inc1: {scd_args.w_inc1}, w_inc2: {scd_args.w_inc2}')

        if scd_args.batch_increase_iter and (epoch + 1) % scd_args.batch_increase_iter == 0:
            train_loader.nrows *= 2
            print(f'nrows: {train_loader.nrows}')
        # if scd_args.adaptive_loss_epoch:
        #     criterion = scd_args.criterion(kind='balanced')
        #     if scd_args.adaptive_loss_epoch < epoch:
        #         criterion = scd_args.criterion(kind='combined')
        with torch.no_grad():

            # update Final layer
            # p = iter(train_loader)
            # data, target = p.next()
            data, target = train_loader.next()
            # for batch_idx, (data, target) in enumerate(test_loader):
            if use_cuda:
                data, target = data.to(device=device, dtype=dtype), target.to(device=device)


            # initial bias
            if epoch == 0 and not scd_args.resume:
                print(f'Randomly initialization based on {scd_args.init} distribution')
                init_bias(net, data)

            # generate adversaries
            adv_data = attacker.perturb((data + 1) / 2.0 if scd_args.divmean else data,
                                        target, 'mean', True)
            adv_data = adv_data.type_as(data)
            data = torch.cat([
                data,
                adv_data * 2 - 1 if scd_args.divmean else adv_data
            ], dim=0)
            target = torch.cat([target, target], dim=0)

            for layer_index, layer in enumerate(layers):
                if len(layers) - layer_index <= scd_args.freeze_layer:
                    # print(f'skip {layer} in training')
                    continue
                # print(layer)
                if 'fc' in layer:
                    if scd_args.no_bias == 2 or scd_args.no_bias and layer_index != 0:
                        update_mid_layer_fc_nobias(net, layers, layer_index, data, dtype,
                                            scd_args, criterion, target, device,
                                                   len(layers)-layer_index == scd_args.bnn_layer)
                    else:
                        update_mid_layer_fc(net, layers, layer_index, data, dtype,
                                            scd_args, criterion, target, device,
                                            len(layers)-layer_index == scd_args.bnn_layer)
                    # continue
                elif 'conv' in layer:
                    # continue
                    if scd_args.no_bias == 2 or scd_args.no_bias and layer_index != 0:
                        update_mid_layer_conv_nobias(net, layers, layer_index, data,
                            dtype, scd_args, criterion, target, device,
                            len(layers)-layer_index == scd_args.bnn_layer)
                    else:
                        update_mid_layer_conv(net, layers, layer_index, data,
                                    dtype, scd_args, criterion, target, device)


            end_time = time.time() - start_time
            # print('This epoch training cost {:.3f} seconds'.format(end_time))
            if (epoch + 1) % scd_args.verbose_iter == 0:
                val_acc, val_loss = evaluation(val_loader, use_cuda, device,
                                     dtype, net, 'Train', criterion)
                if val_acc > best_acc and epoch > 2000:
                    best_acc = val_acc
                    print('Save new best acc: ', best_acc)
                    best_model.load_state_dict(net.state_dict())
                test_acc, test_loss = evaluation(test_loader, use_cuda, device,
                                      dtype, net, 'Test', criterion)

                if scd_args.record:
                    temp_features = get_features(test_loader, use_cuda, device,
                                          dtype, net, 'Test')
                    if epoch == 0:
                        previous_features = temp_features
                        features = {}
                        for layer in layers:
                            features[layer] = []
                    else:
                        for layer in layers:
                            tf = temp_features[layer]
                            pf = previous_features[layer]
                            diff = (tf.view((tf.size(0), -1)) == pf.view((pf.size(0), -1))).float().mean(dim=1)

                            features[layer].append(diff.numpy())
                        previous_features = temp_features


                temp_row = pd.Series(
                    {'epoch': epoch + 1,
                     'train acc': val_acc,
                     'test acc': test_acc,
                     'train loss': val_loss,
                     'test loss': test_loss,
                     }
                )

                if scd_args.save:
                    if epoch == 0:
                        with open(log_file_name + '.temp', 'w') as f:
                            f.write('epoch, train_acc, test_acc, train_loss, test_loss\n')
                    else:
                        with open(log_file_name + '.temp', 'a') as f:
                            f.write(f'{epoch+1}, {val_acc}, {test_acc}, {val_loss}, {test_loss}\n')


                df = df.append(temp_row, ignore_index=True)

            if scd_args.temp_save_per_iter and (epoch + 1) % scd_args.temp_save_per_iter == 0:
                if not os.path.exists(scd_args.save_path):
                    os.makedirs(scd_args.save_path)
                torch.save(best_model.cpu().state_dict(),
                           os.path.join(scd_args.save_path,
                                        scd_args.target + f'_iter#{epoch+1}') + '.pt'
                           )
                print(f"Save {scd_args.target + f'_iter#{epoch+1}' + '.pt'} successfully")

    df.to_csv(log_file_name + '.csv', index=False)
    if scd_args.save:
        if not os.path.exists(scd_args.save_path):
            os.makedirs(scd_args.save_path)
        torch.save(best_model.cpu().state_dict(),
            os.path.join(scd_args.save_path, scd_args.target) + '.pt'
        )
        print(f"Save {scd_args.target + '.pt'} successfully")
        if scd_args.record:
            for name in scd_args.logs.keys():
                # scd_args.logs[name] = np.stack(scd_args.logs[name], axis=0)
                if scd_args.save:
                    dt = pd.DataFrame(scd_args.logs[name])
                    dt.to_csv(log_file_name + f'_{name}.csv', index=False)
            for layer in layers:
                fl = np.stack(features[layer], axis=1)
                dfs = {'data_index': np.arange(fl.shape[0])}
                for i in range(fl.shape[1]):
                    dfs[f'ep{i+1}'] = fl[:, i]

                pd.DataFrame(dfs).to_csv(log_file_name + f'_{layer}_features.csv', index=False)

    return best_model.cpu(), best_acc


if __name__ == '__main__':
    et, vc = print_title()
    scd_args = ModelArgs()

    scd_args.nrows = 1000 / 50000
    scd_args.nfeatures = 1
    scd_args.w_inc = 0.17
    scd_args.tol = 0.00000
    scd_args.local_iter = 1
    scd_args.num_iters = 10000
    scd_args.interval = 10
    scd_args.rounds = 1
    scd_args.w_inc1 = 0.075
    scd_args.updated_fc_features = 128
    scd_args.updated_conv_features = 32
    scd_args.n_jobs = 1
    scd_args.num_gpus = 1
    scd_args.adv_train = False
    scd_args.eps = 0.1
    scd_args.w_inc2 = 0.1
    scd_args.hidden_nodes = 20
    scd_args.evaluation = True
    scd_args.verbose = True
    scd_args.b_ratio = 0.2
    scd_args.cuda = True
    scd_args.seed = 2018
    scd_args.target = 'toy3rrr100'
    scd_args.source = 'cifar10_toy3rrr100_adaptivebs_sign_i1_mce_b1000_lrc0.1_lrf0.1_nb2_nw0_dm1_upc1_upf1_ucf32_fp16_0'
    scd_args.save = False
    scd_args.resume = False
    scd_args.loss = 'mce'
    scd_args.criterion = criterion[scd_args.loss]
    scd_args.structure = arch['toy3rrr100']
    scd_args.dataset = 'cifar10'
    scd_args.num_classes = 10
    scd_args.gpu = 0
    scd_args.fp16 = True
    scd_args.act = 'relu'
    scd_args.updated_fc_nodes = 1
    scd_args.updated_conv_nodes = 1
    scd_args.width = 100
    scd_args.normal_noise = True
    scd_args.verbose = True
    scd_args.normalize = False
    scd_args.batch_size = 256
    scd_args.sigmoid = False
    scd_args.softmax = True if scd_args.loss == 'mce' else False
    scd_args.percentile = True
    scd_args.fail_count = 1
    scd_args.diversity = False
    scd_args.fc_diversity = False
    scd_args.conv_diversity = False
    scd_args.updated_conv_features_diversity = 3
    scd_args.diversity_train_stop_iters = 3000
    scd_args.updated_fc_ratio = 1
    scd_args.updated_conv_ratio = 5
    scd_args.init = 'normal'
    scd_args.logs = {}
    scd_args.no_bias = 2
    scd_args.record = False
    scd_args.scale = 1
    scd_args.save_path = os.path.join('../experiments/checkpoints', 'pt')
    scd_args.divmean = 1
    scd_args.cnn = 1
    scd_args.verbose_iter = 20
    scd_args.freeze_layer = 0
    scd_args.temp_save_per_iter = 0
    scd_args.lr_decay_iter = 0
    scd_args.batch_increase_iter = 3000
    scd_args.aug = 0
    scd_args.balanced_sampling = 1
    scd_args.bnn_layer = 0
    scd_args.epsilon = 0.314
    scd_args.step = 10
    scd_args.alpha = 0.00784



    train_data, test_data, train_label, test_label = load_data('cifar10', scd_args.num_classes)
    if scd_args.cnn:
        train_data = train_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
        test_data = test_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
    if scd_args.divmean:
        train_data = train_data / 0.5 - 1
        test_data = test_data / 0.5 - 1
    np.random.seed(scd_args.seed)



    best_model, val_acc = train_single_cnn01(
        scd_args, None, None, (train_data, test_data, train_label, test_label))






