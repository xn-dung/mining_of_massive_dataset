import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TaxiVolumeDataset(Dataset):
    def __init__(self, npy_path, seq_len=6, pred_len=1, crop_size=7):
        """
        Args:
            npy_path: Đường dẫn tới file .npy đã xử lý.
            seq_len: Số bước thời gian dùng làm đầu vào (Ví dụ: 6 bước = 3 tiếng).
            pred_len: Số bước thời gian cần dự đoán (Ví dụ: 1 bước = 30 phút tới).
            crop_size: Kích thước vùng không gian muốn cắt (Ví dụ: 7 -> cắt 7x7).
        """
        self.data = np.load(npy_path)
        
        self.data = torch.FloatTensor(self.data)
        
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
