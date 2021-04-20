from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from utils.model_utils import calc_emd, calc_cd


# Number of children per tree levels for 2048 output points
tree_arch = {}
tree_arch[2] = [32, 64]
tree_arch[4] = [4, 8, 8, 8]
tree_arch[6] = [2, 4, 4, 4, 4, 4]
tree_arch[8] = [2, 2, 2, 2, 2, 4, 4, 4]


def get_arch(nlevels, npts):
    logmult = int(math.log2(npts/2048))
    assert 2048*(2**(logmult)) == npts, "Number of points is %d, expected 2048x(2^n)" % (npts)
    arch = deepcopy(tree_arch[nlevels])
    while logmult > 0:
        last_min_pos = np.where(arch==np.min(arch))[0][-1]
        arch[last_min_pos]*=2
        logmult -= 1
    return arch


class MLP(nn.Module):
    def __init__(self, dims,  bn=None):
        super().__init__()
        self.model = nn.Sequential()
        for i, num_channels in enumerate(dims[:-2]):
            self.model.add_module('fc_%d' % (i+1), nn.Linear(num_channels, dims[i+1]))
        self.bn = bn
        if self.bn:
            self.batch_norm = nn.BatchNorm1d(dims[-2])

        self.output_layer = nn.Linear(dims[-2], dims[-1])

    def forward(self, features):
        features = self.model(features)
        if self.bn:
            features = self.batch_norm(features)
        features = F.relu(features)
        outputs = self.output_layer(features)
        return outputs


class MLPConv(nn.Module):
    def __init__(self, dims, bn=None):
        super().__init__()
        self.model = nn.Sequential()
        for i, num_channels in enumerate(dims[:-2]):
            self.model.add_module('conv1d_%d' % (i+1), nn.Conv1d(num_channels, dims[i+1], kernel_size=1))
        self.bn = bn
        if self.bn:
            self.batch_norm = nn.BatchNorm1d(dims[-2])

        self.output_layer = nn.Conv1d(dims[-2], dims[-1], kernel_size=1)

    def forward(self, inputs):
        inputs = self.model(inputs)
        if self.bn:
            self.batch_norm.cuda()
            inputs = self.batch_norm(inputs)
        inputs = F.relu(inputs)
        outputs = self.output_layer(inputs)
        return outputs


class CreateLevel(nn.Module):
    def __init__(self, level, input_channels, output_channels, bn, tarch):
        super().__init__()
        self.output_channels = output_channels
        self.mlp_conv = MLPConv([input_channels, input_channels, int(input_channels / 2), int(input_channels / 4),
                                 int(input_channels / 8), output_channels * int(tarch[level])], bn=bn)

    def forward(self, inputs):
        features = self.mlp_conv(inputs)
        features = features.view(features.shape[0], self.output_channels, -1)
        return features


class PCNEncoder(nn.Module):
    def __init__(self, embed_size=1024):
        super().__init__()
        self.conv1 = MLPConv([3, 128, 256])  # no bn
        self.conv2 = MLPConv([512, 512, embed_size])  # no bn

    def forward(self, inputs):
        '''
        :param inputs: B * C * N
        :return: B * C
        '''
        features = self.conv1(inputs)  # [32, 256, 2048]
        features_global, _ = torch.max(features, 2, keepdim=True)  # [32, 256, 1]
        features_global_tiled = features_global.repeat(1, 1, inputs.shape[2])  # [32, 256, 2048]
        features = torch.cat([features, features_global_tiled], dim=1)  # [32, 512, 2048]
        features = self.conv2(features)  # [32, 1024, 2048]
        features, _ = torch.max(features, 2)  # [32, 1024]
        return features


class TopnetDecoder(nn.Module):
    def __init__(self, npts):
        super().__init__()
        self.tarch = get_arch(6, npts)
        self.N = int(np.prod([int(k) for k in self.tarch]))
        assert self.N == npts, "Number of tree outputs is %d, expected %d" % (self.N, npts)
        self.NFEAT = 8
        self.CODE_NFTS = 1024
        self.Nin = self.NFEAT + self.CODE_NFTS
        self.Nout = self.NFEAT
        self.N0 = int(self.tarch[0])
        self.nlevels = len(self.tarch)
        self.mlp = MLP([1024, 256, 64, self.NFEAT * self.N0], bn=True)
        self.mlp_conv_list = nn.ModuleList()
        bn = True
        for i in range(1, self.nlevels):
            if i == self.nlevels - 1:
                self.Nout = 3
                bn = False
            self.mlp_conv_list.append(CreateLevel(i, self.Nin, self.Nout, bn, self.tarch))

    def forward(self, code):
        level0 = self.mlp(code) #
        level0 = torch.tanh(level0)
        level0 = level0.view(-1, self.NFEAT, self.N0)  # (32, 8, 2)
        outs = [level0, ]
        for i in range(self.nlevels-1):
            inp = outs[-1]
            y = torch.unsqueeze(code, dim=2)  # (32, 1024, 1)
            y = y.repeat(1, 1, inp.shape[2])  # (32, 1024, 2)
            y = torch.cat([inp, y], dim=1)  # (32, 1032, 2)
            conv_outs = self.mlp_conv_list[i](y)
            outs.append(torch.tanh(conv_outs))
        return outs[-1]


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.encoder = PCNEncoder()
        self.decoder = TopnetDecoder(args.num_points)
        self.train_loss = args.loss

    def forward(self, x, gt, is_training=True, mean_feature=None, alpha=None):
        features = self.encoder(x)
        out = self.decoder(features)
        out = out.transpose(1, 2).contiguous()

        if is_training:
            if self.train_loss == 'emd':
                loss = calc_emd(out, gt)
            elif self.train_loss == 'cd':
                loss, _ = calc_cd(out, gt)
            else:
                raise NotImplementedError('Train loss is either CD or EMD!')

            return out, loss, loss.mean()
        else:
            emd = calc_emd(out, gt, eps=0.004, iterations=3000)
            cd_p, cd_t, f1 = calc_cd(out, gt, calc_f1=True)
            return {'out1': None, 'out2': out, 'emd': emd, 'cd_p': cd_p, 'cd_t': cd_t, 'f1': f1}
