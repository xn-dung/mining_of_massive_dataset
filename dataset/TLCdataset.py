import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np

class TLCdataset(Dataset):
    def __init__(self, raw_npy, topo_data, context_data, seq_len=8, pred_len=1, crop_size=9, min_val=None, max_val=None):
        raw_data = raw_npy
        
        self.min_val = min_val
        self.max_val = max_val
        normalized_data = (raw_data - self.min_val) / (self.max_val - self.min_val + 1e-7)
        
        self.data = torch.FloatTensor(normalized_data)
        self.topo_data = topo_data
        self.context_data = context_data
        
        self.T, self.C, self.H, self.W = self.data.shape
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.crop_size = crop_size
        if self.crop_size % 2 == 0:
            raise ValueError("crop_size must be odd so the target cell can stay centered.")

        self.pad = self.crop_size // 2

        self.valid_times = self.T - self.seq_len - self.pred_len + 1
        self.cells_per_time = self.H * self.W

    def __len__(self):
        return self.valid_times * self.cells_per_time

    def __getitem__(self, idx):
        time_idx = idx // self.cells_per_time
        cell_idx = idx % self.cells_per_time

        center_h = cell_idx // self.W
        center_w = cell_idx % self.W

        X_full = self.data[time_idx : time_idx + self.seq_len]
        Y_full = self.data[time_idx + self.seq_len : time_idx + self.seq_len + self.pred_len]

        X_full = F.pad(X_full, (self.pad, self.pad, self.pad, self.pad), mode="constant", value=0)
        h_start = center_h
        w_start = center_w
        X_crop = X_full[:, :, h_start : h_start + self.crop_size, w_start : w_start + self.crop_size]

        Y_center = Y_full[:,0:2, center_h, center_w]

        node_id = center_h * self.W + center_w
        topo_vector = self.topo_data[node_id]
        context_seq = self.context_data[time_idx : time_idx + self.seq_len]

        return X_crop, Y_center, topo_vector,context_seq

    def denormalize(self, x):
        c_min = torch.tensor(self.min_val[:, 0:2, 0, 0], dtype=x.dtype, device=x.device)
        c_max = torch.tensor(self.max_val[:, 0:2, 0, 0], dtype=x.dtype, device=x.device)
        return x * (c_max - c_min + 1e-7) + c_min
