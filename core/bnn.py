import tensorflow as tf
import larq as lq
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

tf.keras.backend.set_image_data_format('channels_first')

import torch
import torch.nn as nn
import torch.nn.functional as F


class BNN(object):
    def __init__(self, path):
        if isinstance(path, list):
            self.models = [tf.keras.models.load_model(i) for i in path]
            self.best_model = self.models[0]
        elif isinstance(path, str):
            self.best_model = tf.keras.models.load_model(path)
            self.models = [tf.keras.models.load_model(path)]


    def predict(self, x, best_index=None, batch_size=None):
        data = x.transpose((0, 2, 3, 1))
        if best_index is not None:
            yp = self.models[best_index].predict(data).argmax(axis=1)
            return yp
        else:
            if batch_size:
                n_batch = data.shape[0] // batch_size
                n_rest = data.shape[0] % batch_size
                yps = []
                for j in range(n_batch):
                    yp = []
                    for i in range(len(self.models)):
                        yp.append(self.models[i].predict(data[j*batch_size:(j+1)*batch_size]))
                    yp = np.stack(yp, axis=2)
                    yps.append(yp)

                if n_rest > 0:
                    yp = []
                    for i in range(len(self.models)):
                        yp.append(self.models[i].predict(data[n_batch * batch_size:]))
                    yp = np.stack(yp, axis=2)
                    yps.append(yp)
                yp = np.concatenate(yps, axis=0)
            else:
                yp = []
                for i in range(len(self.models)):
                    yp.append(self.models[i].predict(data))
                yp = np.stack(yp, axis=2)
                
        return yp.mean(axis=2).argmax(axis=1)

    def predict_proba(self, data):
        yp = []
        for i in range(len(self.models)):
            yp.append(self.models[i].predict(data))
        yp = np.stack(yp, dim=2)
        return yp.mean(axis=2)


class Toy3BNN(nn.Module):
    def __init__(self, num_classes=10):
        super(Toy3BNN, self).__init__()
        self.conv1_si = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.conv2_si = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.conv3_si = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.fc4_si = nn.Linear(1024, 100, bias=True)
        self.fc5_si = nn.Linear(100, num_classes, bias=True)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm1d(100)
        self.bn5 = nn.BatchNorm1d(10)
        
    def forward(self, x):
        out = self.conv1_si(x)
        out = F.avg_pool2d(out, 2)
        out = self.bn1(out)
        out = torch.sign(out)
        out = self.conv2_si(out)
        out = F.avg_pool2d(out, 2)
        out = self.bn2(out)
        out = torch.sign(out)
        out = self.conv3_si(out)
        out = F.avg_pool2d(out, 2)
        out = self.bn3(out)
        out = torch.sign(out)
        out = out.reshape((out.size(0), -1))
        out = self.fc4_si(out)
        out = self.bn4(out)
        out = torch.sign(out)
        out = self.fc5_si(out)
        out = self.bn5(out)
        out = F.softmax(out, dim=-1)

        return out


class ModelWrapper2(object):
    def __init__(self, votes, path,):
        self.net = {}
        self.votes = votes
        for i in range(votes):
            self.net[i] = path[i]
            self.net[i].eval()


    def predict(self, x, batch_size=2000):

        if batch_size:
            n_batch = x.shape[0] // batch_size
            n_rest = x.shape[0] % batch_size
            yp = []
            for i in range(n_batch):
                # print(i)
                yp.append(
                    self.inference(x[batch_size * i: batch_size * (i + 1)]))
            if n_rest > 0:
                yp.append(self.inference(x[batch_size * n_batch:]))
            yp = torch.cat(yp, dim=0)

        else:
            # yp = self.net(x)
            yp = self.inference(x)

        return yp.cpu().numpy()


    def predict_proba(self, x, batch_size=None, votes=None):
        if type(x) is not torch.Tensor:
            x = torch.from_numpy(x).type_as(
                self.net[0]._modules[list(self.net[0]._modules.keys())[0]].weight)
        if torch.cuda.is_available():
            for i in range(self.votes):
                self.net[i].cuda()
            x = x.cuda()

        yp = []
        for i in range(self.votes):
            yp.append(self.net[i](x))
        yp = torch.stack(yp, dim=1)

        return yp.mean(dim=1).cpu()

    def inference(self, x, prob=False, all=False, votes=None):
        if type(x) is not torch.Tensor:
            x = torch.from_numpy(x).type_as(
                self.net[0]._modules[list(self.net[0]._modules.keys())[0]].weight)
        if torch.cuda.is_available():
            for i in range(self.votes):
                self.net[i].cuda()
            x = x.cuda()

        yp = []
        for i in range(self.votes):
            yp.append(self.net[i](x))
        yp = torch.stack(yp, dim=1)

        return yp.mean(dim=1).argmax(dim=-1).cpu()