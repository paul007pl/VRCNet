import torch
import os

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir, "utils/Pointnet2.PyTorch/pointnet2"))
import pointnet2_utils as pn2

from utils.gridding import Gridding, GriddingReverse
from utils.cubic_feature_sampling import CubicFeatureSampling
from utils.model_utils import calc_cd, calc_emd


class RandomPointSampling(torch.nn.Module):
    def __init__(self, n_points):
        super(RandomPointSampling, self).__init__()
        self.n_points = n_points

    def forward(self, pred_cloud, partial_cloud=None):
        if partial_cloud is not None:
            pred_cloud = torch.cat([partial_cloud, pred_cloud], dim=1)

        _ptcloud = torch.split(pred_cloud, 1, dim=0)
        ptclouds = []
        for p in _ptcloud:
            non_zeros = torch.sum(p, dim=2).ne(0)
            p = p[non_zeros].unsqueeze(dim=0)
            n_pts = p.size(1)
            if n_pts < self.n_points:
                rnd_idx = torch.cat([torch.randint(0, n_pts, (self.n_points, ))])
            else:
                rnd_idx = torch.randperm(p.size(1))[:self.n_points]
            ptclouds.append(p[:, rnd_idx, :])

        return torch.cat(ptclouds, dim=0).contiguous()


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.num_points = args.num_points
        self.gridding = Gridding(scale=64)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.fc5 = torch.nn.Sequential(
            torch.nn.Linear(16384, 2048),
            torch.nn.ReLU()
        )
        self.fc6 = torch.nn.Sequential(
            torch.nn.Linear(2048, 16384),
            torch.nn.ReLU()
        )
        self.dconv7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.dconv8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.dconv9 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.dconv10 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(1),
            torch.nn.ReLU()
        )
        self.gridding_rev = GriddingReverse(scale=64)
        self.point_sampling = RandomPointSampling(n_points=2048)
        self.feature_sampling = CubicFeatureSampling()
        self.fc11 = torch.nn.Sequential(
            torch.nn.Linear(1792, 1792),
            torch.nn.ReLU()
        )
        self.fc12 = torch.nn.Sequential(
            torch.nn.Linear(1792, 448),
            torch.nn.ReLU()
        )
        self.fc13 = torch.nn.Sequential(
            torch.nn.Linear(448, 112),
            torch.nn.ReLU()
        )
        self.fc14 = torch.nn.Linear(112, 24)

    def forward(self, partial_cloud, gt, is_training=True, mean_feature=None, alpha=None):
        pt_features_64_l = self.gridding(partial_cloud).view(-1, 1, 64, 64, 64)
        pt_features_32_l = self.conv1(pt_features_64_l)
        pt_features_16_l = self.conv2(pt_features_32_l)
        pt_features_8_l = self.conv3(pt_features_16_l)
        pt_features_4_l = self.conv4(pt_features_8_l)
        features = self.fc5(pt_features_4_l.view(-1, 16384))
        pt_features_4_r = self.fc6(features).view(-1, 256, 4, 4, 4) + pt_features_4_l
        pt_features_8_r = self.dconv7(pt_features_4_r) + pt_features_8_l
        pt_features_16_r = self.dconv8(pt_features_8_r) + pt_features_16_l
        pt_features_32_r = self.dconv9(pt_features_16_r) + pt_features_32_l
        pt_features_64_r = self.dconv10(pt_features_32_r) + pt_features_64_l
        sparse_cloud = self.gridding_rev(pt_features_64_r.squeeze(dim=1))
        sparse_cloud = self.point_sampling(sparse_cloud, partial_cloud)
        point_features_32 = self.feature_sampling(sparse_cloud, pt_features_32_r).view(-1, 2048, 256)
        point_features_16 = self.feature_sampling(sparse_cloud, pt_features_16_r).view(-1, 2048, 512)
        point_features_8 = self.feature_sampling(sparse_cloud, pt_features_8_r).view(-1, 2048, 1024)
        point_features = torch.cat([point_features_32, point_features_16, point_features_8], dim=2)
        point_features = self.fc11(point_features)
        point_features = self.fc12(point_features)
        point_features = self.fc13(point_features)
        point_offset = self.fc14(point_features).view(-1, 16384, 3)
        dense_cloud = sparse_cloud.unsqueeze(dim=2).repeat(1, 1, 8, 1).view(-1, 16384, 3) + point_offset
        if self.num_points < 16384:
            idx_fps = pn2.furthest_point_sample(dense_cloud, self.num_points)
            dense_cloud = pn2.gather_operation(dense_cloud, idx_fps)

        if is_training:
            if self.train_loss == 'emd':
                loss1 = calc_emd(sparse_cloud, gt)
                loss2 = calc_emd(dense_cloud, gt)
            elif self.train_loss == 'cd':
                _, loss1 = calc_cd(sparse_cloud, gt)  # cd_t
                _, loss2 = calc_cd(dense_cloud, gt)  # cd_t
            else:
                raise NotImplementedError('Train loss is either CD or EMD!')

            total_train_loss = loss1.mean() + loss2.mean()
            return dense_cloud, loss2, total_train_loss
        else:
            emd = calc_emd(dense_cloud, gt, eps=0.004, iterations=3000)
            cd_p, cd_t, f1 = calc_cd(dense_cloud, gt, calc_f1=True)
            return {'out1': dense_cloud, 'out2': dense_cloud, 'emd': emd, 'cd_p': cd_p, 'cd_t': cd_t, 'f1': f1}