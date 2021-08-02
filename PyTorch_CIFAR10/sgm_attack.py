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
parser.add_argument('--arch', default='resnet18', help='source model',)
parser.add_argument('--source', default='resnet50', help='source model',)
parser.add_argument('--target', default='resnet50', help='source model',)
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
    if correct_index:
        acc = acc[correct_index].mean().item()
    else:
        acc = acc.mean().item()
    print("Accuracy: {:.5f}, "
          "cost {:.2f} seconds".format(acc, time.time() - a))

    return (yp == label).cpu()


def evaluation_cnn01(data_loader, use_cuda, net, correct_index=None):
    a = time.time()
    net.eval()
    correct = 0
    yp = []
    label = []
    for batch_idx, (data, target) in enumerate(data_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data = data.type_as(net._modules[list(net._modules.keys())[0]].weight)
        outputs = net(data)
        # print(outputs.shape)
        yp.append(outputs)
        label.append(target)

    yp = torch.cat(yp, dim=0)
    label = torch.cat(label, dim=0)
    acc = (yp == label).float()
    if correct_index:
        acc = acc[correct_index].mean().item()
    else:
        acc = acc.mean().item()
    print("Accuracy: {:.5f}, "
          "cost {:.2f} seconds".format(acc, time.time() - a))

    return

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
    return torch.cat(advs, dim=0).cpu()


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465])
        self.std = torch.tensor([0.247, 0.243, 0.261])

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :,
                                                                 None, None]

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

    elif 'adv' in source_model:
        model = torchvision.models.resnet18(pretrained=False)
        model._modules['conv1'] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model._modules['fc'] = nn.Linear(512, 10)
        model.load_state_dict(torch.load('checkpoints/cifar10_resnet18_adv_0.pt', map_location='cpu'))

    elif 'nob' in source_model:
        from core.resnet_nob import resnet18
        model = resnet18()
        model.load_state_dict(torch.load('checkpoints/cifar10_resnet18_nob.pt', map_location='cpu'))

    else:
        with open('checkpoints/toy3rrr_bp_fp16_2_32.pkl', 'rb') as f:
            scd = pickle.load(f)
            model = scd.net
            model.float()
        # with open('checkpoints/cifar10_toy3rrr100_bp_adv_32.pkl', 'rb') as f:
        #     scd = pickle.load(f)
        #     model = scd.net
        #     model.float()

    print(f'gamma: {args.gamma}')
    print(f'arch: {args.arch}')
    # model.load_state_dict(torch.load(os.path.join('checkpoints', source_model))['net'])
    if 'normalize1' in source_model:
        model = nn.Sequential(Normalize(), model)
    # model = pretrainedmodels.__dict__[args.arch](num_classes=1000, pretrained='imagenet')
    if use_cuda:
        model.cuda()
    print(f'Source model ("{source_model}") on clean data: ')
    evaluation(test_loader, use_cuda, model)

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
        adversary = MomentumIterativeAttack(predict=model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                            eps=epsilon, nb_iter=args.num_steps, eps_iter=step_size,
                                            decay_factor=args.momentum,
                                            clip_min=0.0, clip_max=1.0, targeted=False)
    else:
        print('using linf PGD attack')
        adversary = LinfPGDAttack(predict=model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                  eps=epsilon, nb_iter=args.num_steps, eps_iter=step_size,
                                  rand_init=False, clip_min=0.0, clip_max=1.0, targeted=False)

    test_adv = generate_adversarial_example(model=model, data_loader=test_loader,
                                 adversary=adversary)
    # advset = TensorDataset(test_adv, torch.LongTensor(testset.targets))
    # np.save('toy3_ensemble_adv.npy', test_adv.cpu().numpy())

    # test_adv = torch.from_numpy(np.load('toy3_ensemble_adv.npy')).cuda()
    advset = TensorDataset(test_adv, torch.LongTensor(testset.targets))
    adv_loader = DataLoader(advset, batch_size=256, shuffle=False, num_workers=0,
                            pin_memory=False)
    print(f'Source model ("{source_model}") on adv data:')
    evaluation(adv_loader, use_cuda, model)

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

    print(f'Target model ("{target_model}") on clean data:')
    evaluation(test_loader, use_cuda, scd)
    print(f'Target model ("{target_model}") on adv data:')
    evaluation(adv_loader, use_cuda, scd)

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

    model_list = [
            'cifar10_toy3bnn_dm1_approx',
            'cifar10_toy3bnn_dm0_approx',
                  ]

    for model in model_list:
        if 'approx' in model:
            from core.bnn import BNN
            net = BNN(
                ['checkpoints/%s_%d.h5' % (model.replace('.pkl', ''), i) for i in
                 range(32)])
        print(f'Model: {model}')
        test_data = testset.data.astype(np.float32) / 255
        adv = test_adv.permute((0, 2, 3, 1))
        if 'dm1' in model:
            test_data = test_data / 0.5 - 1
            adv = adv / 0.5 - 1
        test_label = testset.targets
        yp = net.predict(test_data.astype(np.float32), batch_size=2000)
        print('clean accuracy: ', (yp==test_label).sum()/10000)
        yp = net.predict(adv.cpu().numpy(), batch_size=2000)
        print('adv accuracy: ', (yp==test_label).sum()/10000)

    del model



    # print(f'Target model ("{target_model}") on clean data:')
    # evaluation(test_loader, use_cuda, scd)
    # print(f'Target model ("{target_model}") on adv data:')
    # evaluation(adv_loader, use_cuda, scd)
    #
    target_list = [file for file in os.listdir('checkpoints') if '.pkl' in file]
    target_list = [
        # 'toy3rrr_bp_fp16_32.pkl',
        # 'cifar10_toy3rsss100_adaptivebs_bp_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl',
        # 'cifar10_toy3rrss100_adaptivebs_bp_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl',
        # 'cifar10_toy3ssss100_adaptivebs_bp_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl',
        #
        # 'cifar10_toy3rrr100_bp_adv_32.pkl',
        # 'cifar10_toy3rrss100_adaptivebs_bp_adv_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl',
        # 'cifar10_toy3rsss100_adaptivebs_bp_adv_sign_i1_mce_b1000_lrc0.05_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_32.pkl',
        # 'cifar10_toy3sss100_adaptivebs_sign_i1_mce_b1000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_8.pkl',
        # 'cifar10_toy3rrss100_adaptivebs_bp_sign_i1_01loss_b1000_lrc0.05_lrf0.01_nb2_nw0_dm0_upc1_upf10_ucf32_fp16_32.pkl'
'cifar10_toy3rrss100_bp_sign_i1_01loss_b5000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf10_ucf32_fp16_32.pkl',
'cifar10_toy3rsss100_bp_sign_i1_01loss_b5000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf10_ucf32_fp16_32.pkl',
'cifar10_toy3ssss100_bp_sign_i1_01loss_b5000_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf10_ucf32_fp16_32.pkl',

    ]

    for target_model in target_list:
        try:
            with open(os.path.join('checkpoints', target_model), 'rb') as f:
                scd = pickle.load(f).net
                scd.float()

            # scd.load_state_dict(torch.load(os.path.join('checkpoints', target_model))['net'])
            if 'normalize1' in target_model:
                scd = nn.Sequential(Normalize(), scd)
            if use_cuda:
                scd.cuda()
            print(f'Target model ("{target_model}") on clean data:')
            try:
                evaluation_cnn01(test_loader, use_cuda, scd)
            except:
                evaluation(test_loader, use_cuda, scd)


            # print(f'Source model ("{source_model}") on adv data:')
            # evaluation(adv_loader, use_cuda, model)
            print(f'Target model ("{target_model}") on adv data:')
            try:
                evaluation_cnn01(adv_loader, use_cuda, scd)
            except:
                evaluation(adv_loader, use_cuda, scd)
        except:
            continue