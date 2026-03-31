import torch
from torch.utils.data import Dataset
import numpy as np

class TLCdataset(Dataset):
    def __init__(self, npy_path, seq_len=8, pred_len=1, crop_size=9):
        raw_data = np.load(npy_path) 
        
        self.min_val = raw_data.min(axis=(0, 2, 3), keepdims=True)
        self.max_val = raw_data.max(axis=(0, 2, 3), keepdims=True)
        
        normalized_data = (raw_data - self.min_val) / (self.max_val - self.min_val + 1e-7)
        
        self.data = torch.FloatTensor(normalized_data)
        
        self.T, self.C, self.H, self.W = self.data.shape
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.crop_size = crop_size

        self.valid_times = self.T - self.seq_len - self.pred_len + 1
        self.h_steps = self.H - self.crop_size + 1
        self.w_steps = self.W - self.crop_size + 1
        self.patches_per_time = self.h_steps * self.w_steps

    def __len__(self):
        return self.valid_times * self.patches_per_time

    def __getitem__(self, idx):
        time_idx = idx // self.patches_per_time
        patch_idx = idx % self.patches_per_time
        
        h_start = patch_idx // self.w_steps
        w_start = patch_idx % self.w_steps

        X_full = self.data[time_idx : time_idx + self.seq_len]
        Y_full = self.data[time_idx + self.seq_len : time_idx + self.seq_len + self.pred_len]

        X_crop = X_full[:, :, h_start : h_start + self.crop_size, w_start : w_start + self.crop_size]
        Y_crop = Y_full[:, :, h_start : h_start + self.crop_size, w_start : w_start + self.crop_size]

        return X_crop, Y_crop

    def denormalize(self, x):
        return x * (self.max_val - self.min_val + 1e-7) + self.min_val