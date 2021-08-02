import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys

sys.path.append('..')
from advertorch.attacks import LinfPGDAttack, MomentumIterativeAttack
# import pretrainedmodels
from utils_sgm import register_hook_for_resnet, register_hook_for_densenet
import pickle
# from tools import load_data
import torchvision
import torchvision.transforms as transforms
from cifar10_models import *
import sys

sys.path.append('..')
from core.cnn import Toy3BN

parser = argparse.ArgumentParser(description='PyTorch ImageNet Attack')
parser.add_argument('--input-dir', default='', help='Input directory with images.')
parser.add_argument('--output-dir', default='', help='Output directory with images.')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for adversarial attack')
parser.add_argument('--arch', default='resnet18', help='source model', )
parser.add_argument('--source', default='resnet50', help='source model', )
parser.add_argument('--target', default='resnet50', help='source model', )
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=8, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2, type=float,
                    help='perturb step size')
parser.add_argument('--gamma', default=0.2, type=float)
parser.add_argument('--momentum', default=0.0, type=float)
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--target_attack', type=int, default=0, metavar='N',
                    help='input batch size for adversarial attack')
parser.add_argument('--n_classes', type=int, default=10, metavar='N',
                    help='input batch size for adversarial attack')
args = parser.parse_args()
use_cuda = True


def evaluation(data_loader, use_cuda, net, correct_index=None):
    a = time.time()
    net.eval()
    correct = 0
    yp = []
    label = []
    for batch_idx, (data, target) in enumerate(data_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        outputs = net(data)
        # print(outputs.shape)
        outputs = outputs.max(1)[1]
        yp.append(outputs)
        label.append(target)
        correct += outputs.eq(target).sum().item()

    yp = torch.cat(yp, dim=0)
    label = torch.cat(label, dim=0)
    acc = (yp == label).float()
    if correct_index is not None:
        acc = acc[correct_index].mean().item()
    else:
        acc = acc.mean().item()
    print("Accuracy: {:.5f}, "
          "cost {:.2f} seconds".format(acc, time.time() - a))

    return yp.cpu(), (yp == label).cpu()


def evaluation_cnn01(data_loader, use_cuda, net, correct_index=None):
    a = time.time()
    net.eval()
    correct = 0
    yp = []
    label = []
    for batch_idx, (data, target) in enumerate(data_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # data = data.type_as(net._modules[list(net._modules.keys())[0]].weight)
        outputs = net(data)
        # print(outputs.shape)
        yp.append(outputs)
        label.append(target)

    yp = torch.cat(yp, dim=0)
    label = torch.cat(label, dim=0)
    acc = (yp == label).float()
    if correct_index is not None:
        acc = acc[correct_index].mean().item()
    else:
        acc = acc.mean().item()
    print("Accuracy: {:.5f}, "
          "cost {:.2f} seconds".format(acc, time.time() - a))

    return yp.cpu(), (yp == label).cpu()


def generate_adversarial_example(model, data_loader, adversary):
    """
    generate and save adversarial example
    """
    model.eval()
    advs = []
    labels = []
    for batch_idx, (inputs, target) in enumerate(data_loader):
        if use_cuda:
            inputs = inputs.cuda()
            target = target.cuda()
        # with torch.no_grad():
        #     # _, pred = model(inputs).topk(1, 1, True, True)
        #     pred = model(inputs).argmax(dim=1)

        # craft adversarial images
        if args.target_attack:
            target = (target + 1) % args.n_classes
        inputs_adv = adversary.perturb(inputs, target)
        advs.append(inputs_adv)
        labels.append(target)
        # save adversarial images
    return torch.cat(advs, dim=0).cpu(), torch.cat(labels, dim=0).cpu()


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465])
        self.std = torch.tensor([0.247, 0.243, 0.261])

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :,
                                                                 None, None]

class Dm(nn.Module):
    def __init__(self):
        super(Dm, self).__init__()

    def forward(self, x):
        return x / 0.5 - 1

if __name__ == '__main__':
    # train_data, test_data, train_label, test_label = load_data('cifar10', 10)
    # train_data = train_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
    # test_data = test_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
    # # test_data = np.random.normal(size=(2000, 3, 224, 224)).astype(np.float32)
    # trainset = TensorDataset(torch.from_numpy(train_data.astype(np.float32)),
    #                          torch.from_numpy(train_label.astype(np.int64)))
    # testset = TensorDataset(torch.from_numpy(test_data.astype(np.float32)),
    #                         torch.from_numpy(test_label.astype(np.int64)))
    #
    # test_loader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=0,
    #                          pin_memory=False)
    DATA = 'cifar10'

    if DATA == 'cifar10':
        train_dir = '/home/y/yx277/research/ImageDataset/cifar10'
        test_dir = '/home/y/yx277/research/ImageDataset/cifar10'

    test_transform_list = [transforms.ToTensor()]
    test_transform = transforms.Compose(test_transform_list)
    testset = torchvision.datasets.CIFAR10(root=test_dir, train=False, download=True,
                                           transform=test_transform)

    test_idx = [True if i < args.n_classes else False for i in testset.targets]
    testset.data = testset.data[test_idx]
    testset.targets = [i for i in testset.targets if i < args.n_classes]
    test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False,
                                              num_workers=4, pin_memory=False)

    source_model = args.source
    if 'resnet18' in source_model:
        model = resnet18(pretrained=True)

        # args.gamma = 0.2

    elif 'resnet50' in source_model:
        model = resnet50(pretrained=True)
        args.gamma = 0.4

    elif 'vgg19_bn' in source_model:
        model = vgg19_bn(pretrained=True)
        args.gamma = 0.4

    # elif 'adv' in source_model:
    #     model = torchvision.models.resnet18(pretrained=False)
    #     model._modules['conv1'] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    #     model._modules['fc'] = nn.Linear(512, 10)
    #     model.load_state_dict(
    #         torch.load('checkpoints/cifar10_resnet18_adv_0.pt', map_location='cpu'))

    elif 'nob' in source_model:
        from core.resnet_nob import resnet18

        model = resnet18()
        model.load_state_dict(torch.load('checkpoints/cifar10_resnet18_nob.pt', map_location='cpu'))

    elif 'toy3bn' in source_model:
        with open('checkpoints/toy3bnrrr_bp_fp16_32.pkl', 'rb') as f:
            model = pickle.load(f)
            model = model.net[0]
    else:
        # with open('checkpoints/toy3rrr_bp_fp16_2_32.pkl', 'rb') as f:
        #     scd = pickle.load(f)
        #     model = scd.net
        #     model.float()
        # with open('checkpoints/cifar10_toy3rrr100_bp_adv_32.pkl', 'rb') as f:
        #     scd = pickle.load(f)
        #     model = scd.net
        #     model.float()
        # source_model = 'cifar10_toy3sss100scale_abp_retrain0_sign_i1_mce_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8'
        with open(f'../scd/experiments/checkpoints/{source_model}.pkl', 'rb') as f:
            scd = pickle.load(f)
            model = scd.net
            model.float()
    print(f'gamma: {args.gamma}')
    print(f'arch: {args.arch}')
    print(f'epsilon: {args.epsilon}')
    # model.load_state_dict(torch.load(os.path.join('checkpoints', source_model))['net'])
    if 'normalize1' in source_model:
        model = nn.Sequential(Normalize(), model)
    if 'dm1' in source_model:
        model = nn.Sequential(Dm(), model)
    # model = pretrainedmodels.__dict__[args.arch](num_classes=1000, pretrained='imagenet')
    if use_cuda:
        model.cuda()
    print(f'Source model ("{source_model}") on clean data: ')
    yp, correct_index = evaluation(test_loader, use_cuda, model)

    epsilon = args.epsilon / 255.0
    if args.step_size < 0:
        step_size = epsilon / args.num_steps
    else:
        step_size = args.step_size / 255.5

    if args.gamma < 1.0:
        if args.arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            register_hook_for_resnet(model, arch=args.arch, gamma=args.gamma)
        elif args.arch in ['densenet121', 'densenet169', 'densenet201']:
            register_hook_for_densenet(model, arch=args.arch, gamma=args.gamma)
        else:
            raise ValueError('Current code only supports resnet/densenet. '
                             'You can extend this code to other architectures.')

    if args.momentum > 0.0:
        print('using PGD attack with momentum = {}'.format(args.momentum))
        adversary = MomentumIterativeAttack(predict=model,
                                            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                            eps=epsilon, nb_iter=args.num_steps, eps_iter=step_size,
                                            decay_factor=args.momentum,
                                            clip_min=0.0, clip_max=1.0, targeted=False)
    else:
        print('using linf PGD attack')
        adversary = LinfPGDAttack(
                                    # predict=model,
                                  # loss_fn=nn.CrossEntropyLoss(reduction="mean"),
                                  predict=lambda x: model(x).log(),
                                  loss_fn=nn.NLLLoss(reduction="mean"),
                                  eps=epsilon, nb_iter=args.num_steps, eps_iter=step_size,
                                  rand_init=False, clip_min=0.0, clip_max=1.0, targeted=bool(args.target_attack))

    test_adv, source_adv_target = generate_adversarial_example(model=model, data_loader=test_loader,
                                                               adversary=adversary)
    # if args.target_attack:
    #     np.save('toy3_ensemble_ta_adv.npy', test_adv.cpu().numpy())
    #     np.save('toy3_ensemble_ta_target.npy', source_adv_target.cpu().numpy())
    # else:
    #     np.save('toy3_ensemble_ut_adv.npy', test_adv.cpu().numpy())
    #     np.save('toy3_ensemble_ut_target.npy', source_adv_target.cpu().numpy())

    # if args.target_attack:
    #     test_adv = torch.from_numpy(np.load('toy3_ensemble_ta_adv.npy')).cuda()
    #     source_adv_target = torch.from_numpy(np.load('toy3_ensemble_ta_target.npy'))
    # else:
    #     test_adv = torch.from_numpy(np.load('toy3_ensemble_ut_adv.npy')).cuda()
    #     source_adv_target = torch.from_numpy(np.load('toy3_ensemble_ut_target.npy'))
    advset = TensorDataset(test_adv, torch.LongTensor(testset.targets))
    adv_loader = DataLoader(advset, batch_size=256, shuffle=False, num_workers=0,
                            pin_memory=False)


    if args.target_attack:
        advset_target = TensorDataset(test_adv, source_adv_target)
        adv_target_loader = DataLoader(advset_target, batch_size=256, shuffle=False, num_workers=0,
                                       pin_memory=False)
        print(f'Source model ("{source_model}") match rate on adv target data (all):')
        evaluation(adv_target_loader, use_cuda, model)
        print(f'Source model ("{source_model}") match rate on adv target data (correct):')
        evaluation(adv_target_loader, use_cuda, model, correct_index)
    else:
        print(f'Source model ("{source_model}") on adv data (all):')
        source_adv_yp, _ = evaluation(adv_loader, use_cuda, model)
        print(f'Source model ("{source_model}") on adv data (correct):')
        source_adv_yp, _ = evaluation(adv_loader, use_cuda, model, correct_index)
        
        # print('Generate new adv target dataloader based on predicted label')
        # advset_target = TensorDataset(test_adv, source_adv_yp)
        # adv_target_loader = DataLoader(advset_target, batch_size=256, shuffle=False, num_workers=0,
        #                         pin_memory=False)

    # with open('cifar10_10_resnet18.pt', 'rb') as f:
    #     scd = pickle.load(f)
    target_model = args.target
    if 'resnet18' in target_model:
        scd = resnet18(pretrained=True)
        # args.gamma = 0.2

    elif 'resnet50' in target_model:
        scd = resnet50(pretrained=True)

        # args.gamma = 0.4
    elif 'vgg19_bn' in target_model:
        scd = vgg19_bn(pretrained=True, device='cpu')

    elif 'vgg19' in target_model:
        scd = vgg19(pretrained=True, device='cpu')

    elif 'toy3bn' in target_model:
        with open('checkpoints/toy3bnrrr_bp_fp16_32.pkl', 'rb') as f:
            scd = pickle.load(f)

    if 'normalize1' in target_model:
        scd = nn.Sequential(Normalize(), scd)
    if use_cuda:
        scd.cuda()

    # print(f'Target model ("{target_model}") on clean data:')
    # yp, correct_index = evaluation(test_loader, use_cuda, scd)
    #
    # if args.target_attack:
    #     print(f'Target model ("{target_model}") match rate on adv target data (all):')
    #     evaluation(adv_target_loader, use_cuda, scd)
    #     print(f'Target model ("{target_model}") match rate on adv target data (correct):')
    #     evaluation(adv_target_loader, use_cuda, scd, correct_index)
    # else:
    #     print(f'Target model ("{target_model}") on adv data (all):')
    #     evaluation(adv_loader, use_cuda, scd)
    #     print(f'Target model ("{target_model}") on adv data (correctly):')
    #     evaluation(adv_loader, use_cuda, scd, correct_index)
    # model_list = [
    #         'toy3bnrrr_bp_fp16_32.pkl'
    #               ]
    #
    # for model in model_list:
    #     with open(os.path.join('checkpoints', model), 'rb') as f:
    #         net = pickle.load(f)
    #     print(f'Model: {model}')
    #     test_data = testset.data.astype(np.float32).transpose((0, 3, 1, 2)) / 255
    #     test_label = testset.targets
    #     yp = net.predict(test_data.astype(np.float32), batch_size=500)
    #     print('clean accuracy: ', (yp==test_label).sum()/10000)
    #     yp = net.predict(test_adv.cpu().numpy(), batch_size=500)
    #     print('adv accuracy: ', (yp==test_label).sum()/10000)

    # model_list = [
    #     # 'cifar10_toy3bnn_aug_dm1_approx',
    #     # 'cifar10_toy3bnn_dm0_approx',
    #     'cifar10_toy3srr100ban_dm1_aug1_approx',
    #     'cifar10_toy3ssr100ban_dm1_aug1_approx',
    #     'cifar10_toy3sss100ban_dm1_aug1_approx',
    # ]
    #
    # for model in model_list:
    #     if 'approx' in model:
    #         from core.bnn import BNN
    #
    #         net = BNN(
    #             ['checkpoints/%s_%d.h5' % (model.replace('.pkl', ''), i) for i in
    #              range(8)])
    #     print(f'Model: {model}')
    #     test_data = testset.data.astype(np.float32) / 255
    #     adv = test_adv.permute((0, 2, 3, 1))
    #     if 'dm1' in model:
    #         test_data = test_data / 0.5 - 1
    #         adv = adv / 0.5 - 1
    #     test_label = testset.targets
    #     yp = net.predict(test_data.astype(np.float32), batch_size=2000)
    #     print('clean accuracy: ', (yp == test_label).astype(np.float32).mean())
    #     correct_index = (yp == test_label)
    #     if args.target_attack:
    #         yp = net.predict(adv.cpu().numpy(), batch_size=2000)
    #         print(f'Target model ("{model}") match rate on adv target data (all): ',
    #               (yp == source_adv_target.numpy()).astype(np.float32).mean())
    #
    #         print(f'Target model ("{model}") match rate on adv target data (correct): ',
    #               (yp == source_adv_target.numpy()).astype(np.float32)[correct_index].mean())
    #
    #     else:
    #         yp = net.predict(adv.cpu().numpy(), batch_size=2000)
    #         print(f'Target model ("{model}") on adv data (all): ', (yp == test_label).astype(np.float32).mean())
    #
    #         print(f'Target model ("{model}") on adv data (correctly): ',
    #               (yp == test_label).astype(np.float32)[correct_index].mean())
    #
    # del model


    #
    target_list = [file for file in os.listdir('../scd/experiments/checkpoints') if '.pkl' in file]
    target_list = [
        # 'toy3rrr_bp_fp16_32.pkl',
        # 'cifar10_toy3rsss100_adaptivebs_bp_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl',
        # 'cifar10_toy3rrss100_adaptivebs_bp_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl',
        # 'cifar10_toy3ssss100_adaptivebs_bp_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl',
        #
        # 'cifar10_toy3rrr100_bp_adv_32.pkl',
        # 'cifar10_toy3rrss100_adaptivebs_bp_adv_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl',
        # 'cifar10_toy3rsss100_adaptivebs_bp_adv_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl',
        # 'cifar10_toy3ssss100_adaptivebs_bp_adv_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl',
        # 'cifar10_toy3rrss100_adaptivebs_bnn_bp_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_5.pkl',
        # 'cifar10_toy3ssss100_adaptivebs_bnn_bp_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_6.pkl',
        # 'cifar10_toy3sss100_adaptivebs_sign_i1_mce_b1000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_8.pkl',
        # 'cifar10_toy3rrss100_adaptivebs_bp_sign_i1_01loss_b1000_lrc0.05_lrf0.01_nb2_nw0_dm0_upc1_upf10_ucf32_fp16_32.pkl'
        # 'cifar10_toy3rrss100_bp_sign_i1_01loss_b5000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf10_ucf32_fp16_32.pkl',
        # 'cifar10_toy3rsss100_bp_sign_i1_01loss_b5000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf10_ucf32_fp16_32.pkl',
        # 'cifar10_toy3ssss100_bp_sign_i1_01loss_b5000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf10_ucf32_fp16_32.pkl',
        # 'cifar10_toy3rrr100_sign_i1_mce_b5000_lrc0.075_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_8.pkl',
        # 'toy3rrr_bp_noaug_fp16_4.pkl',
        # 'cifar10_toy3rrr100_sign_i1_mce_b5000_lrc0.075_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_4.pkl',
        # 'cifar10_toy3rrs100_sign_i1_mce_b5000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_8.pkl',
        # 'cifar10_toy3rsr100_sign_i1_mce_b5000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_8.pkl',
        # 'cifar10_toy3rss100_sign_i1_mce_b5000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_8.pkl',
        # 'cifar10_toy3srr100_sign_i1_mce_b5000_lrc0.075_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_8.pkl',
        # 'cifar10_toy3ssr100_sign_i1_mce_b5000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_8.pkl',
        # 'cifar10_toy3sss100_sign_i1_mce_b5000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_8.pkl',
        #
        # 'cifar10_toy3rrr100_adaptivebs_sign_i1_mce_b1000_lrc0.075_lrf0.2_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_4.pkl',
        # 'toy3rrr_bp_noaug_sn_fp16_8.pkl',
        'toy3rrr_bp_noaug_fp16_8.pkl',
        # 'cifar10_toy3rrr100_bp_adv_noaug_8.pkl',
        # 'cifar10_toy3rrr100_adaptivebs_sign_i1_mce_b1000_lrc0.1_lrf0.1_nb2_nw0_dm1_upc1_upf1_ucf32_fp16_8.pkl',
        # 'cifar10_toy3rrs100_adaptivebs_sign_i1_mce_b1000_lrc0.075_lrf0.1_nb2_nw0_dm1_upc1_upf1_ucf32_fp16_8.pkl',
        # 'cifar10_toy3rrss100_adaptivebs_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm1_upc1_upf1_ucf32_fp16_8.pkl',
        # 'cifar10_toy3rrr100_adaptivebs_sign_i1_mce_b1000_lrc0.1_lrf0.1_nb2_nw0_dm1_upc1_upf1_ucf32_fp16_1.pkl'
        # 'cifar10_toy3rrr100_adaptivebs_adv_sign_i1_mce_b500_lrc0.1_lrf0.2_nb2_nw0_dm1_upc1_upf1_ucf32_fp16_8.pkl',
        # 'cifar10_toy3rrs100fs_adaptivebs_sign_i1_mce_b1000_lrc0.075_lrf0.1_nb2_nw0_dm1_upc1_upf1_ucf32_fp16_8.pkl',
        # 'cifar10_toy3rrss100fs_adaptivebs_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm1_upc1_upf1_ucf32_fp16_8.pkl',
        
        # 'cifar10_toy3srr100fs_adaptivebs_bp_sign_i1_mce_b1000_lrc0.075_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_8.pkl',
        # 'cifar10_toy3ssr100fs_adaptivebs_bp_sign_i1_mce_b1000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_8.pkl',
        # 'cifar10_toy3sss100fs_adaptivebs_bp_sign_i1_mce_b1000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_8.pkl',
        # 'cifar10_toy3ssss100fs_adaptivebs_bp_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_8.pkl',
        # 'cifar10_binary_toy3rrr_bp_fp16_8.pkl',
        # 'cifar10_binary_toy3rrs100_adaptivebs_bp_sign_i1_mce_b1000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_8.pkl',
        # 'cifar10_binary_toy3rrss100_adaptivebs_bp_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_8.pkl',
        # 'toy3rrr_bp_fp16_8.pkl',
        # 'cifar10_toy3rrr100_bp_adv_8.pkl',
        # 'cifar10_toy3rrr100_bp_adv_ac_8.pkl',
        # 'cifar10_toy3srr100scale_bp_fp16_8.pkl',
        # 'cifar10_toy3ssr100scale_bp_fp16_8.pkl',
        # 'cifar10_toy3sss100scale_bp_fp16_8.pkl',
        # 'cifar10_toy3ssss100scale_bp_fp16_8.pkl',

        # 'cifar10_toy3srr100_adaptivebs_bp_sign_i1_mce_b1000_lrc0.075_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_8.pkl',
        # 'cifar10_toy3ssr100_adaptivebs_bp_sign_i1_mce_b1000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_8.pkl',
        # 'cifar10_toy3sss100_adaptivebs_bp_sign_i1_mce_b1000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_8.pkl',
        # 'cifar10_toy3ssss100_adaptivebs_bp_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_8.pkl',
        # 'cifar10_toy3srr100scale_adaptivebs_abp_sign_i1_mce_b200_lrc0.075_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_uniform_fp16_8.pkl',
        # 'cifar10_toy3ssr100scale_adaptivebs_abp_sign_i1_mce_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_uniform_fp16_8.pkl',
        # 'cifar10_toy3sss100scale_adaptivebs_abp_sign_i1_mce_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_uniform_fp16_8.pkl',
        # 'cifar10_toy3ssss100scale_adaptivebs_abp_sign_i1_mce_b200_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_uniform_fp16_8.pkl',
        # 'cifar10_toy3srr100scale_adaptivebs_abp_sign_i1_mce_b200_lrc0.075_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',
        # 'cifar10_toy3ssr100scale_adaptivebs_abp_sign_i1_mce_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',
        # 'cifar10_toy3sss100scale_adaptivebs_abp_sign_i1_mce_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',
        # 'cifar10_toy3ssss100scale_adaptivebs_abp_sign_i1_mce_b200_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',
        # 'cifar10_toy3srr100scale_adaptivebs_bnn_abp_sign_i1_mce_b200_lrc0.075_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',
        # 'cifar10_toy3ssr100scale_adaptivebs_bnn_abp_sign_i1_mce_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',
        # 'cifar10_toy3sss100scale_adaptivebs_bnn_abp_sign_i1_mce_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',
        # 'cifar10_toy3ssss100scale_adaptivebs_bnn_abp_sign_i1_mce_b200_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',
        # 'cifar10_toy3srr100scale_abp_sign_i1_mce_b200_lrc0.025_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',
        # 'cifar10_toy3ssr100scale_abp_sign_i1_mce_b200_lrc0.025_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',
        # 'cifar10_toy3sss100scale_abp_sign_i1_mce_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',
        # 'cifar10_toy3ssss100scale_abp_sign_i1_mce_b200_lrc0.05_lrf0.7_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',
        # 'cifar10_toy3ssr100scale_abp_retrain0_sign_i1_mce_b200_lrc0.025_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',
        # 'cifar10_toy3sss100scale_abp_retrain0_sign_i1_mce_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',
        # 'cifar10_toy3ssss100scale_abp_retrain0_sign_i1_mce_b200_lrc0.05_lrf0.7_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',
        # 'cifar10_toy3srr100scale_abp_adv_thm_sign_i1_mce_b200_lrc0.025_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',
        # 'cifar10_toy3ssr100scale_abp_adv_thm_retrain0_sign_i1_mce_b200_lrc0.025_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',
        # 'cifar10_toy3sss100scale_abp_adv_thm_retrain0_sign_i1_mce_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',
        # 'cifar10_toy3ssss100scale_abp_adv_thm_retrain0_sign_i1_mce_b200_lrc0.05_lrf0.7_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',

        # 'cifar10_toy3rrr100ap1_bp_fp16_8.pkl',
        # 'cifar10_toy3rrr100ap2_bp_fp16_8.pkl',
        # 'cifar10_binary_toy3rrr_bp_fp16_8.pkl',
        # 'cifar10_binary_toy3rrr100_bp_adv_8.pkl',
        # 'cifar10_binary_toy3srr100scale_abp_sign_i1_mce_b200_lrc0.025_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',
        # 'cifar10_binary_toy3ssr100scale_abp_sign_i1_mce_b200_lrc0.025_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',
        # 'cifar10_binary_toy3sss100scale_abp_sign_i1_mce_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',
        # 'cifar10_binary_toy3ssss100scale_abp_sign_i1_mce_b200_lrc0.05_lrf0.7_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',
        # 'cifar10_binary_toy3ssr100scale_abp_retrain0_sign_i1_mce_b200_lrc0.025_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',
        # 'cifar10_binary_toy3sss100scale_abp_retrain0_sign_i1_mce_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',
        # 'cifar10_binary_toy3ssss100scale_abp_retrain0_sign_i1_mce_b200_lrc0.05_lrf0.7_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8.pkl',

    ]

    for target_model in target_list:
        try:
            with open(os.path.join('../scd/experiments/checkpoints', target_model), 'rb') as f:
                scd = pickle.load(f).net
                scd.float()

            # scd.load_state_dict(torch.load(os.path.join('checkpoints', target_model))['net'])
            if 'normalize1' in target_model:
                scd = nn.Sequential(Normalize(), scd)
            if 'dm1' in target_model:
                scd = nn.Sequential(Dm(), scd)
            if use_cuda:
                scd.cuda()
            print(f'Target model ("{target_model}") on clean data:')
            try:
                yp, correct_index = evaluation_cnn01(test_loader, use_cuda, scd)
            except:
                yp, correct_index = evaluation(test_loader, use_cuda, scd)

            # print(f'Source model ("{source_model}") on adv data:')
            # evaluation(adv_loader, use_cuda, model)
            if args.target_attack:
                try:
                    print(f'Target model ("{target_model}") match rate on adv target data (all):')
                    evaluation_cnn01(adv_target_loader, use_cuda, scd)
                    print(f'Target model ("{target_model}") match rate on adv target data (correct):')
                    evaluation_cnn01(adv_target_loader, use_cuda, scd, correct_index)
                except:
                    evaluation(adv_target_loader, use_cuda, scd)
                    print(f'Target model ("{target_model}") match rate on adv target data (correct):')
                    evaluation(adv_target_loader, use_cuda, scd, correct_index)
            else:
                try:
                    print(f'Target model ("{target_model}") on adv data (all):')
                    evaluation_cnn01(adv_loader, use_cuda, scd)
                    print(f'Target model ("{target_model}") on adv data (correctly):')
                    evaluation_cnn01(adv_loader, use_cuda, scd, correct_index)
                except:
                    print(f'Target model ("{target_model}") on adv data (all):')
                    evaluation(adv_loader, use_cuda, scd)
                    print(f'Target model ("{target_model}") on adv data (correctly):')
                    evaluation(adv_loader, use_cuda, scd, correct_index)
        except:
            continue