import torch
import numpy as np
import torch.utils.data as data
import h5py
import os


class ShapeNetH5(data.Dataset):
    def __init__(self, train=True, npoints=2048, use_mean_feature=0):
        # train data only has input(2048) and gt(2048)
        self.npoints = npoints
        self.train = train
        self.use_mean_feature = use_mean_feature
        proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if train:
            self.input_path = os.path.join(proj_dir, "data/cascade/train_data.h5")
        else:
            self.input_path = os.path.join(proj_dir, "data/cascade/test_data.h5")
        input_file = h5py.File(self.input_path, 'r')
        self.input_data = input_file['incomplete_pcds'][()]
        self.labels = input_file['labels'][()]
        self.gt_data = input_file['complete_pcds'][()]
        input_file.close()

        if self.use_mean_feature == 1:
            self.mean_feature_path = os.path.join(proj_dir, "data/cascade/mean_feature.h5")
            mean_feature_file = h5py.File(self.mean_feature_path, 'r')
            self.mean_feature = mean_feature_file['mean_features'][()]
            mean_feature_file.close()

        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy(np.array(self.input_data[index])).float()
        complete = torch.from_numpy(np.array(self.gt_data[index])).float()
        label = torch.from_numpy(np.array(self.labels[index])).int()
        if self.use_mean_feature == 1:
            mean_feature_input = torch.from_numpy(np.array(self.mean_feature[label])).float()
            return label, partial, complete, mean_feature_input
        else:
            return label, partial, complete
