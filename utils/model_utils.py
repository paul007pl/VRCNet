import torch
import math
import os
import sys
import torch.nn as nn
import torch.nn.functional as F



proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir, "utils/Pointnet2.PyTorch/pointnet2"))
import pointnet2_utils as pn2
sys.path.append(os.path.join(proj_dir, "utils/emd"))
import emd_module as emd
sys.path.append(os.path.join(proj_dir, "utils/ChamferDistancePytorch"))
from chamfer3D import dist_chamfer_3D
from fscore import fscore


class EF_expansion(nn.Module):
    def __init__(self, input_size, output_size=64, step_ratio=2, k=4):
        super(EF_expansion, self).__init__()
        self.step_ratio = step_ratio
        self.k = k
        self.input_size = input_size
        self.output_size = output_size

        self.conv1 = nn.Conv2d(input_size * 2, output_size, 1)
        self.conv2 = nn.Conv2d(input_size * 2 + output_size, output_size * step_ratio, 1)
        self.conv3 = nn.Conv2d(output_size, output_size, 1)

    def forward(self, x):
        batch_size, _, num_points = x.size()

        input_edge_feature = get_graph_feature(x, self.k, minus_center=False).permute(0, 1, 3,
                                                                                      2).contiguous()  # B C K N
        edge_feature = self.conv1(input_edge_feature)
        edge_feature = F.relu(torch.cat((edge_feature, input_edge_feature), 1))

        edge_feature = F.relu(self.conv2(edge_feature))  # B C K N
        edge_feature = edge_feature.permute(0, 2, 3, 1).contiguous().view(batch_size, self.k,
                                                                          num_points * self.step_ratio,
                                                                          self.output_size).permute(0, 3, 1,
                                                                                                    2)  # B C K N

        edge_feature = self.conv3(edge_feature)
        edge_feature, _ = torch.max(edge_feature, 2)

        return edge_feature


def attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)  # B x 4 x points x points
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


def calc_cd(output, gt, calc_f1=False):
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, _, _ = cham_loss(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2)
        return cd_p, cd_t, f1
    else:
        return cd_p, cd_t


def calc_emd(output, gt, eps=0.005, iterations=50):
    emd_loss = emd.emdModule()
    dist, _ = emd_loss(output, gt, eps, iterations)
    emd_out = torch.sqrt(dist).mean(1)
    return emd_out


def edge_preserve_sampling(feature_input, point_input, num_samples, k=10):
    batch_size = feature_input.size()[0]
    feature_size = feature_input.size()[1]
    num_points = feature_input.size()[2]

    p_idx = pn2.furthest_point_sample(point_input, num_samples)
    point_output = pn2.gather_operation(point_input.transpose(1, 2).contiguous(), p_idx).transpose(1,
                                                                                                   2).contiguous()

    pk = int(min(k, num_points))
    _, pn_idx = knn_point(pk, point_input, point_output)
    pn_idx = pn_idx.detach().int()
    neighbor_feature = pn2.gather_operation(feature_input, pn_idx.view(batch_size, num_samples * pk)).view(batch_size,
                                                                                                           feature_size,
                                                                                                           num_samples,
                                                                                                           pk)
    neighbor_feature, _ = torch.max(neighbor_feature, 3)

    center_feature = pn2.grouping_operation(feature_input, p_idx.unsqueeze(2)).view(batch_size, -1, num_samples)

    net = torch.cat((center_feature, neighbor_feature), 1)

    return net, p_idx, pn_idx, point_output


def get_edge_features(x, idx):
    batch_size, num_points, k = idx.size()
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    x = x.squeeze(2)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims).permute(0, 3, 2, 1)  # B, C, K, N
    return feature


def gen_grid(num_grid_point):
    x = torch.linspace(-0.05, 0.05, steps=num_grid_point)
    x, y = torch.meshgrid(x, x)
    grid = torch.stack([x, y], axis=-1).view(2, num_grid_point ** 2)
    return grid


def gen_1d_grid(num_grid_point):
    x = torch.linspace(-0.05, 0.05, num_grid_point)
    grid = x.view(1, num_grid_point)
    return grid


def gen_grid_up(up_ratio, grid_size=0.2):
    sqrted = int(math.sqrt(up_ratio)) + 1
    for i in range(1, sqrted + 1).__reversed__():
        if (up_ratio % i) == 0:
            num_x = i
            num_y = up_ratio // i
            break

    grid_x = torch.linspace(-grid_size, grid_size, steps=num_x)
    grid_y = torch.linspace(-grid_size, grid_size, steps=num_y)

    x, y = torch.meshgrid(grid_x, grid_y)  # x, y shape: (2, 1)
    grid = torch.stack([x, y], dim=-1).view(-1, 2).transpose(0, 1).contiguous()
    return grid


def get_graph_feature(x, k=20, minus_center=True):
    idx = knn(x, k=k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    if minus_center:
        feature = torch.cat((x, feature - x), dim=3).permute(0, 3, 1, 2)
    else:
        feature = torch.cat((x, feature), dim=3).permute(0, 3, 1, 2)
    return feature


def get_repulsion_loss(pred, nsample=20, radius=0.07):
    # pred: (batch_size, npoint,3)
    # idx = pn2.ball_query(radius, nsample, pred, pred)
    idx = knn(pred.transpose(1, 2).contiguous(), nsample).int()
    pred_flipped = pred.transpose(1, 2).contiguous()
    grouped_pred = pn2.grouping_operation(pred_flipped, idx)  # (B, C, npoint, nsample)
    grouped_pred -= pred_flipped.unsqueeze(-1)

    # get the uniform loss
    h = 0.03
    dist_square = torch.sum(grouped_pred ** 2, dim=1)
    dist_square, idx = torch.topk(-dist_square, 5)
    dist_square = -dist_square[:, :, 1:]  # remove the first one
    dist_square = torch.max(torch.FloatTensor([1e-12]).expand_as(dist_square).cuda(), dist_square)
    dist = torch.sqrt(dist_square)
    weight = torch.exp(-dist_square / h ** 2)
    uniform_loss = torch.mean(radius - dist * weight)
    return uniform_loss


def get_uniform_loss(pcd, percentages=[0.004, 0.006, 0.008, 0.010, 0.012], radius=1.0):
    B, N, C = pcd.size()
    npoint = int(N * 0.05)
    loss = 0
    for p in percentages:
        nsample = int(N*p)
        r = math.sqrt(p*radius)
        disk_area = math.pi * (radius ** 2) * p/nsample
        new_xyz = pn2.gather_operation(pcd.transpose(1, 2).contiguous(),
                                       pn2.furthest_point_sample(pcd, npoint)).transpose(1, 2).contiguous()
        idx = pn2.ball_query(r, nsample, pcd, new_xyz)
        expect_len = math.sqrt(disk_area)

        grouped_pcd = pn2.grouping_operation(pcd.transpose(1,2).contiguous(), idx)
        grouped_pcd = grouped_pcd.permute(0, 2, 3, 1).contiguous().view(-1, nsample, 3)

        var, _ = knn_point(2, grouped_pcd, grouped_pcd)
        uniform_dis = -var[:, :, 1:]

        uniform_dis = torch.sqrt(torch.abs(uniform_dis+1e-8))
        uniform_dis = torch.mean(uniform_dis, dim=-1)
        uniform_dis = ((uniform_dis - expect_len)**2 / (expect_len + 1e-8))

        mean = torch.mean(uniform_dis)
        mean = mean*math.pow(p*100,2)
        loss += mean
    return loss/len(percentages)


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def knn_point(pk, point_input, point_output):
    m = point_output.size()[1]
    n = point_input.size()[1]

    inner = -2 * torch.matmul(point_output, point_input.transpose(2, 1).contiguous())
    xx = torch.sum(point_output ** 2, dim=2, keepdim=True).repeat(1, 1, n)
    yy = torch.sum(point_input ** 2, dim=2, keepdim=False).unsqueeze(1).repeat(1, m, 1)
    pairwise_distance = -xx - inner - yy
    dist, idx = pairwise_distance.topk(k=pk, dim=-1)
    return dist, idx


def knn_point_all(pk, point_input, point_output):
    m = point_output.size()[1]
    n = point_input.size()[1]

    inner = -2 * torch.matmul(point_output, point_input.transpose(2, 1).contiguous())
    xx = torch.sum(point_output ** 2, dim=2, keepdim=True).repeat(1, 1, n)
    yy = torch.sum(point_input ** 2, dim=2, keepdim=False).unsqueeze(1).repeat(1, m, 1)
    pairwise_distance = -xx - inner - yy
    dist, idx = pairwise_distance.topk(k=pk, dim=-1)

    return dist, idx


def symmetric_sample(points, num=512):
    p1_idx = pn2.furthest_point_sample(points, num)
    input_fps = pn2.gather_operation(points.transpose(1, 2).contiguous(), p1_idx).transpose(1, 2).contiguous()
    x = torch.unsqueeze(input_fps[:, :, 0], dim=2)
    y = torch.unsqueeze(input_fps[:, :, 1], dim=2)
    z = torch.unsqueeze(-input_fps[:, :, 2], dim=2)
    input_fps_flip = torch.cat([x, y, z], dim=2)
    input_fps = torch.cat([input_fps, input_fps_flip], dim=1)
    return input_fps


def three_nn_upsampling(target_points, source_points):
    dist, idx = pn2.three_nn(target_points, source_points)
    dist = torch.max(dist, torch.ones(1).cuda() * 1e-10)
    norm = torch.sum((1.0 / dist), 2, keepdim=True)
    norm = norm.repeat(1, 1, 3)
    weight = (1.0 / dist) / norm

    return idx, weight
