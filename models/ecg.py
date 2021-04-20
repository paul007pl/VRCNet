from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from utils.model_utils import *
from models.pcn import PCN_encoder

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir, "utils/Pointnet2.PyTorch/pointnet2"))
import pointnet2_utils as pn2


class Stack_conv(nn.Module):
    def __init__(self, input_size, output_size, act=None):
        super(Stack_conv, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module('conv', nn.Conv2d(input_size, output_size, 1))

        if act is not None:
            self.model.add_module('act', act)

    def forward(self, x):
        y = self.model(x)
        y = torch.cat((x, y), 1)
        return y


class Dense_conv(nn.Module):
    def __init__(self, input_size, growth_rate=64, dense_n=3, k=16):
        super(Dense_conv, self).__init__()
        self.growth_rate = growth_rate
        self.dense_n = dense_n
        self.k = k
        self.comp = growth_rate * 2
        self.input_size = input_size

        self.first_conv = nn.Conv2d(self.input_size * 2, growth_rate, 1)

        self.input_size += self.growth_rate

        self.model = nn.Sequential()
        for i in range(dense_n - 1):
            if i == dense_n - 2:
                self.model.add_module('stack_conv_%d' % (i + 1), Stack_conv(self.input_size, self.growth_rate, None))
            else:
                self.model.add_module('stack_conv_%d' % (i + 1),
                                      Stack_conv(self.input_size, self.growth_rate, nn.ReLU()))
                self.input_size += growth_rate

    def forward(self, x):
        y = get_graph_feature(x, k=self.k)
        y = F.relu(self.first_conv(y))
        y = torch.cat((y, x.unsqueeze(3).repeat(1, 1, 1, self.k)), 1)

        y = self.model(y)
        y, _ = torch.max(y, 3)
        return y


class EF_encoder(nn.Module):
    def __init__(self, growth_rate=24, dense_n=3, k=16, hierarchy=[1024, 256, 64], input_size=3, output_size=256):
        super(EF_encoder, self).__init__()
        self.growth_rate = growth_rate
        self.comp = growth_rate * 2
        self.dense_n = dense_n
        self.k = k
        self.hierarchy = hierarchy

        self.init_channel = 24

        self.conv1 = nn.Conv1d(input_size, self.init_channel, 1)
        self.dense_conv1 = Dense_conv(self.init_channel, self.growth_rate, self.dense_n, self.k)

        out_channel_size_1 = (self.init_channel * 2 + self.growth_rate * self.dense_n)  # 24*2 + 24*3 = 120
        self.conv2 = nn.Conv1d(out_channel_size_1 * 2, self.comp, 1)
        self.dense_conv2 = Dense_conv(self.comp, self.growth_rate, self.dense_n, self.k)

        out_channel_size_2 = (
                    out_channel_size_1 * 2 + self.comp + self.growth_rate * self.dense_n)  # 120*2 + 48 + 24*3 = 360
        self.conv3 = nn.Conv1d(out_channel_size_2 * 2, self.comp, 1)
        self.dense_conv3 = Dense_conv(self.comp, self.growth_rate, self.dense_n, self.k)

        out_channel_size_3 = (
                    out_channel_size_2 * 2 + self.comp + self.growth_rate * self.dense_n)  # 360*2 + 48 + 24*3 = 840
        self.conv4 = nn.Conv1d(out_channel_size_3 * 2, self.comp, 1)
        self.dense_conv4 = Dense_conv(self.comp, self.growth_rate, self.dense_n, self.k)

        out_channel_size_4 = out_channel_size_3 * 2 + self.comp + self.growth_rate * self.dense_n  # 840*2 + 48 + 24*3 = 1800
        self.gf_conv = nn.Conv1d(out_channel_size_4, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 1024)

        out_channel_size = out_channel_size_4 + 1024
        self.conv5 = nn.Conv1d(out_channel_size, 1024, 1)

        out_channel_size = out_channel_size_3 + 1024
        self.conv6 = nn.Conv1d(out_channel_size, 768, 1)

        out_channel_size = out_channel_size_2 + 768
        self.conv7 = nn.Conv1d(out_channel_size, 512, 1)

        out_channel_size = out_channel_size_1 + 512
        self.conv8 = nn.Conv1d(out_channel_size, output_size, 1)

    def forward(self, x):
        point_cloud1 = x[:, 0:3, :]
        point_cloud1 = point_cloud1.transpose(1, 2).contiguous()

        x0 = F.relu(self.conv1(x))  # 24
        x1 = F.relu(self.dense_conv1(x0))  # 24 + 24 * 3 = 96
        x1 = torch.cat((x1, x0), 1)  # 120
        x1d, _, _, point_cloud2 = edge_preserve_sampling(x1, point_cloud1, self.hierarchy[0], self.k)  # 240

        x2 = F.relu(self.conv2(x1d))  # 48
        x2 = F.relu(self.dense_conv2(x2))  # 48 + 24 * 3 = 120
        x2 = torch.cat((x2, x1d), 1)  # 120 + 240 = 360
        x2d, _, _, point_cloud3 = edge_preserve_sampling(x2, point_cloud2, self.hierarchy[1], self.k)  # 720

        x3 = F.relu(self.conv3(x2d))
        x3 = F.relu(self.dense_conv3(x3))
        x3 = torch.cat((x3, x2d), 1)
        x3d, _, _, point_cloud4 = edge_preserve_sampling(x3, point_cloud3, self.hierarchy[2], self.k)

        x4 = F.relu(self.conv4(x3d))
        x4 = F.relu(self.dense_conv4(x4))
        x4 = torch.cat((x4, x3d), 1)

        global_feat = self.gf_conv(x4)
        global_feat, _ = torch.max(global_feat, -1)
        global_feat = F.relu(self.fc1(global_feat))
        global_feat = F.relu(self.fc2(global_feat)).unsqueeze(2).repeat(1, 1, self.hierarchy[2])

        x4 = torch.cat((global_feat, x4), 1)
        x4 = F.relu(self.conv5(x4))
        idx, weight = three_nn_upsampling(point_cloud3, point_cloud4)
        x4 = pn2.three_interpolate(x4, idx, weight)

        x3 = torch.cat((x3, x4), 1)
        x3 = F.relu(self.conv6(x3))
        idx, weight = three_nn_upsampling(point_cloud2, point_cloud3)
        x3 = pn2.three_interpolate(x3, idx, weight)

        x2 = torch.cat((x2, x3), 1)
        x2 = F.relu(self.conv7(x2))
        idx, weight = three_nn_upsampling(point_cloud1, point_cloud2)
        x2 = pn2.three_interpolate(x2, idx, weight)

        x1 = torch.cat((x1, x2), 1)
        x1 = self.conv8(x1)
        return x1


class ECG_decoder(nn.Module):
    def __init__(self, num_coarse, num_fine, num_input):
        super(ECG_decoder, self).__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine

        self.scale = int(np.ceil(num_fine / (num_coarse + num_input)))

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_coarse * 3)

        self.dense_feature_size = 256
        self.expand_feature_size = 64
        self.input_size = 3

        self.encoder = EF_encoder(growth_rate=24, dense_n=3, k=16, hierarchy=[1024, 256, 64],
                                  input_size=self.input_size, output_size=self.dense_feature_size)

        if self.scale >= 2:
            self.expansion = EF_expansion(input_size=self.dense_feature_size, output_size=self.expand_feature_size,
                                          step_ratio=self.scale, k=4)
            self.conv1 = nn.Conv1d(self.expand_feature_size, self.expand_feature_size, 1)
        else:
            self.expansion = None
            self.conv1 = nn.Conv1d(self.dense_feature_size, self.expand_feature_size, 1)
        self.conv2 = nn.Conv1d(self.expand_feature_size, 3, 1)

    def forward(self, global_feat, point_input):
        batch_size = global_feat.size()[0]
        coarse = F.relu(self.fc1(global_feat))
        coarse = F.relu(self.fc2(coarse))
        coarse = self.fc3(coarse).view(batch_size, 3, self.num_coarse)
        org_points_input = point_input
        points = torch.cat((coarse, org_points_input), 2)

        dense_feat = self.encoder(points)

        if self.scale >= 2:
            dense_feat = self.expansion(dense_feat)

        point_feat = F.relu(self.conv1(dense_feat))
        fine = self.conv2(point_feat)

        num_out = fine.size()[2]
        if num_out > self.num_fine:
            fine = pn2.gather_operation(fine,
                                        pn2.furthest_point_sample(fine.transpose(1, 2).contiguous(), self.num_fine))

        return coarse, fine


class Model(nn.Module):
    def __init__(self, args, num_coarse=1024, num_input=2048):
        super(Model, self).__init__()
        self.num_coarse = num_coarse
        self.num_points = args.num_points
        self.train_loss = args.loss
        self.encoder = PCN_encoder()
        self.decoder = ECG_decoder(num_coarse, self.num_points, num_input)

    def forward(self, x, gt, is_training=True, mean_feature=None, alpha=None):
        if mean_feature:
            raise NotImplementedError
        feat = self.encoder(x)
        out1, out2 = self.decoder(feat, x)
        out1 = out1.transpose(1, 2).contiguous()
        out2 = out2.transpose(1, 2).contiguous()
        uniform_loss1 = get_uniform_loss(out1)
        uniform_loss2 = get_uniform_loss(out2)

        if is_training:
            if self.train_loss == 'emd':
                loss1 = calc_emd(out1, gt)
                loss2 = calc_emd(out2, gt)
            elif self.train_loss == 'cd':
                loss1, _ = calc_cd(out1, gt)
                loss2, _ = calc_cd(out2, gt)
            else:
                raise NotImplementedError('Train loss is either CD or EMD!')

            total_train_loss = loss1.mean() + uniform_loss1.mean() * 0.1 + \
                               (loss2.mean() + uniform_loss2.mean() * 0.1) * alpha
            return out2, loss2, total_train_loss
        else:
            emd = calc_emd(out2, gt, eps=0.004, iterations=3000)
            cd_p, cd_t, f1 = calc_cd(out2, gt, calc_f1=True)
            return {'out1': out1, 'out2': out2, 'emd': emd, 'cd_p': cd_p, 'cd_t': cd_t, 'f1': f1}
