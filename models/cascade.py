from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import sys
import os
from utils.model_utils import calc_emd, calc_cd, gen_grid_up

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir, "utils/Pointnet2.PyTorch/pointnet2"))
import pointnet2_utils as pn2
from pointnet2_modules import PointnetSAModuleMSG


def symmetric_sample(points, num):
    p1_idx = pn2.furthest_point_sample(points, num)
    input_fps = pn2.gather_operation(points.transpose(1, 2).contiguous(), p1_idx).transpose(1, 2).contiguous()
    x = torch.unsqueeze(input_fps[:, :, 0], dim=2)
    y = torch.unsqueeze(input_fps[:, :, 1], dim=2)
    z = torch.unsqueeze(-input_fps[:, :, 2], dim=2)
    input_fps_flip = torch.cat([x, y, z], dim=2)
    input_fps = torch.cat([input_fps, input_fps_flip], dim=1)
    return input_fps


class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.model = nn.Sequential()
        for i, num_channels in enumerate(dims[:-1]):
            self.model.add_module('fc_%d' % (i+1), nn.Linear(num_channels, dims[i+1]))
            if i != len(dims) - 2:
                self.model.add_module('relu_%d' % (i+1), nn.ReLU())
   
    def forward(self, features):
        return self.model(features)


class MLPConv(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.model = nn.Sequential()
        for i, num_channels in enumerate(dims[:-1]):
            self.model.add_module('conv1d_%d' % (i+1), nn.Conv1d(num_channels, dims[i+1], kernel_size=1))
            if i != len(dims) - 2:
                self.model.add_module('relu_%d' % (i+1), nn.ReLU())

    def forward(self, inputs):
        return self.model(inputs)


class ContractExpandOperation(nn.Module):
    def __init__(self, num_input_channels, up_ratio):
        super().__init__()
        self.up_ratio = up_ratio
        # PyTorch default padding is 'VALID'
        # !!! rmb to add in L2 loss for conv2d weights
        self.conv2d_1 = nn.Conv2d(num_input_channels, 64, kernel_size=(1, self.up_ratio), stride=(1, 1))
        self.conv2d_2 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d_3 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, inputs):  # (32, 64, 2048)
        net = inputs.view(inputs.shape[0], inputs.shape[1], self.up_ratio, -1)  # (32, 64, 2, 1024)
        net = net.permute(0, 1, 3, 2).contiguous()  # (32, 64, 1024, 2)
        net = F.relu(self.conv2d_1(net))  # (32, 64, 1024, 1)
        net = F.relu(self.conv2d_2(net))  # (32, 128, 1024, 1)
        net = net.permute(0, 2, 3, 1).contiguous()  # (32, 1024, 1, 128)
        net = net.view(net.shape[0], -1, self.up_ratio, 64)  # (32, 1024, 2, 64)
        net = net.permute(0, 3, 1, 2).contiguous()  # (32, 64, 1024, 2)
        net = F.relu(self.conv2d_3(net)) # (32, 64, 1024, 2)
        net = net.view(net.shape[0], 64, -1)  # (32, 64, 2048)
        return net


class Encoder(nn.Module):
    def __init__(self, embed_size=1024):
        super().__init__()
        self.conv1 = MLPConv([3, 128, 256])
        self.conv2 = MLPConv([512, 512, embed_size])

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


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.coarse_mlp = MLP([1024, 1024, 1024, 512 * 3])
        self.mean_fc = nn.Linear(1024, 128)
        self.up_branch_mlp_conv_mf = MLPConv([1157, 128, 64])
        self.up_branch_mlp_conv_nomf = MLPConv([1029, 128, 64])
        self.contract_expand = ContractExpandOperation(64, 2)
        self.fine_mlp_conv = MLPConv([64, 512, 512, 3])

    def forward(self, code, inputs, step_ratio, num_extract=512, mean_feature=None):
        '''
        :param code: B * C
        :param inputs: B * C * N
        :param step_ratio: int
        :param num_extract: int
        :param mean_feature: B * C
        :return: coarse(B * N * C), fine(B, N, C)
        '''
        coarse = torch.tanh(self.coarse_mlp(code))  # (32, 1536)
        coarse = coarse.view(-1, 512, 3)  # (32, 512, 3)
        coarse = coarse.transpose(2, 1).contiguous()  # (32, 3, 512)

        inputs_new = inputs.transpose(2, 1).contiguous()  # (32, 2048, 3)
        input_fps = symmetric_sample(inputs_new, int(num_extract/2))  # [32, 512,  3]
        input_fps = input_fps.transpose(2, 1).contiguous()  # [32, 3, 512]
        level0 = torch.cat([input_fps, coarse], 2)   # (32, 3, 1024)
        if num_extract > 512:
            level0_flipped = level0.transpose(2, 1).contiguous()
            level0 = pn2.gather_operation(level0, pn2.furthest_point_sample(level0_flipped, 1024))

        for i in range(int(math.log2(step_ratio))):
            num_fine = 2 ** (i + 1) * 1024
            grid = gen_grid_up(2 ** (i + 1)).cuda().contiguous()
            grid = torch.unsqueeze(grid, 0)   # (1, 2, 2)
            grid_feat = grid.repeat(level0.shape[0], 1, 1024)   # (32, 2, 2048)
            point_feat = torch.unsqueeze(level0, 3).repeat(1, 1, 1, 2)  # (32, 3, 1024, 2)
            point_feat = point_feat.view(-1, 3, num_fine)  # (32, 3, 2048)
            global_feat = torch.unsqueeze(code, 2).repeat(1, 1, num_fine)  # (32, 1024, 2048)

            if mean_feature is not None:
                mean_feature_use = F.relu(self.mean_fc(mean_feature))  #(32, 128)
                mean_feature_use = torch.unsqueeze(mean_feature_use, 2).repeat(1, 1, num_fine)  #(32, 128, 2048)
                feat = torch.cat([grid_feat, point_feat, global_feat, mean_feature_use], dim=1)  # (32, 1157, 2048)
                feat1 = F.relu(self.up_branch_mlp_conv_mf(feat))  # (32, 64, 2048)
            else:
                feat = torch.cat([grid_feat, point_feat, global_feat], dim=1)
                feat1 = F.relu(self.up_branch_mlp_conv_nomf(feat))  # (32, 64, 2048)

            feat2 = self.contract_expand(feat1) # (32, 64, 2048)
            feat = feat1 + feat2  # (32, 64, 2048)

            fine = self.fine_mlp_conv(feat) + point_feat  # (32, 3, 2048)
            level0 = fine

        return coarse.transpose(1, 2).contiguous(), fine.transpose(1, 2).contiguous()


class Discriminator(nn.Module):
    def __init__(self, args, divide_ratio=2):
        super(Discriminator, self).__init__()
        self.num_points = args.num_points
        self.pointnet_sa_module = PointnetSAModuleMSG(npoint=int(self.num_points/8), radii=[0.1, 0.2, 0.4], nsamples=[16, 32, 128],
                                                      mlps=[[3,  32 // divide_ratio, 32 // divide_ratio, 64 // divide_ratio],
                                                       [3,  64 // divide_ratio, 64 // divide_ratio, 128 // divide_ratio],
                                                       [3, 64 // divide_ratio, 96 // divide_ratio, 128 // divide_ratio]],)
        self.patch_mlp_conv = MLPConv([(64//divide_ratio + 128 // divide_ratio + 128 // divide_ratio), 1])

    def forward(self, xyz):
        _, l1_points = self.pointnet_sa_module(xyz, features=None)
        patch_values = self.patch_mlp_conv(l1_points)
        return patch_values


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.step_ratio = 2 * (args.num_points // 2048)
        self.train_loss = args.loss
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, gt, is_training=True, mean_feature=None, alpha=None):
        features_partial_pt = self.encoder(x)
        out1, out2 = self.decoder(features_partial_pt, x, self.step_ratio, mean_feature=mean_feature)

        if is_training:
            if self.train_loss == 'emd':
                loss1 = calc_emd(out1, gt)
                loss2 = calc_emd(out2, gt)
            elif self.train_loss == 'cd':
                loss1, _ = calc_cd(out1, gt)
                loss2, _ = calc_cd(out2, gt)
            total_train_loss = loss1.mean() + loss2.mean() * alpha
            return out2, loss2, total_train_loss
        else:
            emd = calc_emd(out2, gt, eps=0.004, iterations=3000)
            cd_p, cd_t, f1 = calc_cd(out2, gt, calc_f1=True)
            return {'out1': out1, 'out2': out2, 'emd': emd, 'cd_p': cd_p, 'cd_t': cd_t, 'f1': f1}


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
