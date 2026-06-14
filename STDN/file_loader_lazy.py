import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _resolve_path(path):
    return Path(path).expanduser().resolve()


def ensure_npy_array(array_path, key, cache_dir=None):
    """
    Return a .npy path that can be opened with np.load(..., mmap_mode="r").

    npz files cannot be sliced lazily by numpy. To keep RAM low, this function
    extracts the inner key.npy member to disk once, then all later reads use
    memory mapping.
    """
    array_path = _resolve_path(array_path)
    if array_path.suffix == ".npy":
        return array_path
    if array_path.suffix != ".npz":
        raise ValueError(f"Unsupported array file type: {array_path}")

    cache_dir = _resolve_path(cache_dir) if cache_dir is not None else array_path.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy_path = cache_dir / f"{array_path.stem}_{key}.npy"

    if npy_path.exists() and npy_path.stat().st_mtime >= array_path.stat().st_mtime:
        return npy_path

    member_name = f"{key}.npy"
    tmp_path = npy_path.with_suffix(".tmp.npy")
    if tmp_path.exists():
        tmp_path.unlink()

    with zipfile.ZipFile(array_path, "r") as zf:
        if member_name not in zf.namelist():
            raise KeyError(f"{array_path} does not contain {member_name}. Available: {zf.namelist()}")
        with zf.open(member_name, "r") as src, tmp_path.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)

    tmp_path.replace(npy_path)
    return npy_path


def collate_fn(batch):
    has_context = len(batch[0]) == 9
    if has_context:
        att_cnn, att_flow, att_lstm, cnn, flow, lstm, label, context, att_context = zip(*batch)
    else:
        att_cnn, att_flow, att_lstm, cnn, flow, lstm, label = zip(*batch)

    att_cnn_stacked = [torch.stack([sample[i] for sample in att_cnn]) for i in range(len(att_cnn[0]))]
    att_cnn_stacked = [x.permute(0, 3, 1, 2) for x in att_cnn_stacked]

    att_flow_stacked = [torch.stack([sample[i] for sample in att_flow]) for i in range(len(att_flow[0]))]
    att_flow_stacked = [x.permute(0, 3, 1, 2) for x in att_flow_stacked]

    att_lstm_stacked = [torch.stack([sample[i] for sample in att_lstm]) for i in range(len(att_lstm[0]))]

    cnn_stacked = [torch.stack([sample[i] for sample in cnn]) for i in range(len(cnn[0]))]
    cnn_stacked = [x.permute(0, 3, 1, 2) for x in cnn_stacked]

    flow_stacked = [torch.stack([sample[i] for sample in flow]) for i in range(len(flow[0]))]
    flow_stacked = [x.permute(0, 3, 1, 2) for x in flow_stacked]

    lstm_stacked = torch.stack(lstm)
    label_stacked = torch.stack(label)

    if not has_context:
        return att_cnn_stacked, att_flow_stacked, att_lstm_stacked, cnn_stacked, flow_stacked, lstm_stacked, label_stacked

    context_stacked = torch.stack(context)
    att_context_stacked = torch.stack(att_context)

    return (
        att_cnn_stacked,
        att_flow_stacked,
        att_lstm_stacked,
        cnn_stacked,
        flow_stacked,
        lstm_stacked,
        label_stacked,
        context_stacked,
        att_context_stacked,
    )


class STDNLazyDataset(Dataset):
    """
    Lazy PyTorch Dataset for STDN.

    It keeps only raw volume/flow arrays open as memory maps and builds one
    sample inside __getitem__. This avoids materializing all STDN features in
    RAM before training.
    """

    def __init__(
        self,
        datatype,
        config_path="data.json",
        att_lstm_num=3,
        long_term_lstm_seq_len=3,
        short_term_lstm_seq_len=7,
        hist_feature_daynum=7,
        last_feature_num=None,
        nbhd_size=1,
        cnn_nbhd_size=3,
        cache_dir=None,
    ):
        if datatype not in {"train", "test"}:
            raise ValueError("datatype must be 'train' or 'test'")
        if long_term_lstm_seq_len % 2 != 1:
            raise ValueError("Attention LSTM sequence length must be odd")

        self.datatype = datatype
        self.config_path = Path(config_path)
        with self.config_path.open("r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.timeslot_daynum = int(86400 / self.config["timeslot_sec"])
        self.threshold = int(self.config["threshold"])
        self.att_lstm_num = att_lstm_num
        self.long_term_lstm_seq_len = long_term_lstm_seq_len
        self.short_term_lstm_seq_len = short_term_lstm_seq_len
        self.hist_feature_daynum = hist_feature_daynum
        self.last_feature_num = last_feature_num if last_feature_num is not None else self.timeslot_daynum
        self.nbhd_size = nbhd_size
        self.cnn_nbhd_size = cnn_nbhd_size

        volume_key = f"volume_{datatype}"
        flow_key = f"flow_{datatype}"
        self.volume_max = float(self.config["volume_train_max"])
        self.flow_max = float(self.config["flow_train_max"])

        volume_path = ensure_npy_array(self.config[volume_key], "volume", cache_dir=cache_dir)
        flow_path = ensure_npy_array(self.config[flow_key], "flow", cache_dir=cache_dir)

        self.volume = np.load(volume_path, mmap_mode="r")
        self.flow = np.load(flow_path, mmap_mode="r")
        self.context = None
        self.context_time = None
        self.context_feature_names = []
        self.context_dim = 0

        context_key = f"context_{datatype}"
        if context_key in self.config:
            context_npz = np.load(self.config[context_key], allow_pickle=True)
            self.context = np.asarray(context_npz["context"], dtype=np.float32)
            self.context_time = np.asarray(context_npz["time"]).astype(str) if "time" in context_npz.files else None
            if "feature_names" in context_npz.files:
                self.context_feature_names = np.asarray(context_npz["feature_names"]).astype(str).tolist()
            self.context_dim = int(self.context.shape[1])

            volume_npz = np.load(self.config[volume_key], allow_pickle=True)
            flow_npz = np.load(self.config[flow_key], allow_pickle=True)
            volume_time = np.asarray(volume_npz["time"]).astype(str) if "time" in volume_npz.files else None
            flow_time = np.asarray(flow_npz["time"]).astype(str) if "time" in flow_npz.files else None

            if self.context_time is not None:
                if volume_time is not None and not np.array_equal(volume_time, self.context_time):
                    raise ValueError("volume time and context time are not aligned")
                if flow_time is not None and not np.array_equal(flow_time, self.context_time):
                    raise ValueError("flow time and context time are not aligned")

        if self.volume.ndim != 4:
            raise ValueError(f"Expected volume shape (T,H,W,C), got {self.volume.shape}")
        if self.flow.ndim != 6:
            raise ValueError(f"Expected flow shape (2,T,H,W,H,W), got {self.flow.shape}")

        self.time_count = int(self.volume.shape[0])
        self.grid_h = int(self.volume.shape[1])
        self.grid_w = int(self.volume.shape[2])
        self.volume_type = int(self.volume.shape[3])
        self.cell_count = self.grid_h * self.grid_w

        self.time_start = (
            (self.hist_feature_daynum + self.att_lstm_num) * self.timeslot_daynum
            + self.long_term_lstm_seq_len
        )
        self.time_end = self.time_count
        self.target_time_count = max(0, self.time_end - self.time_start)
        self.n_samples = self.target_time_count * self.cell_count

        self.feature_vec_len = (
            self.hist_feature_daynum * self.volume_type
            + self.last_feature_num * self.volume_type
            + (2 * self.nbhd_size + 1) * (2 * self.nbhd_size + 1) * self.volume_type
        )

    def __len__(self):
        return self.n_samples

    def sample_shape_info(self):
        return {
            "n_samples": self.n_samples,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "grid_h": self.grid_h,
            "grid_w": self.grid_w,
            "volume_type": self.volume_type,
            "feature_vec_len": self.feature_vec_len,
            "cnn_nbhd_dim": 2 * self.cnn_nbhd_size + 1,
            "context_dim": self.context_dim,
            "context_feature_names": self.context_feature_names,
        }

    def _decode_index(self, idx):
        if idx < 0 or idx >= self.n_samples:
            raise IndexError(idx)
        time_offset = idx // self.cell_count
        cell = idx % self.cell_count
        t = self.time_start + time_offset
        x = cell // self.grid_w
        y = cell % self.grid_w
        return int(t), int(x), int(y)

    def _volume_at(self, *index):
        # Data is already normalized in NPZ files, no need to divide again
        return np.asarray(self.volume[index], dtype=np.float32)

    def _flow_at(self, *index):
        # Data is already normalized in NPZ files, no need to divide again
        return np.asarray(self.flow[index], dtype=np.float32)

    def _local_volume(self, real_t, x, y, radius):
        size = 2 * radius + 1
        feature = np.zeros((size, size, self.volume_type), dtype=np.float32)

        x0 = max(0, x - radius)
        x1 = min(self.grid_h, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(self.grid_w, y + radius + 1)

        out_x0 = x0 - (x - radius)
        out_y0 = y0 - (y - radius)
        feature[out_x0:out_x0 + (x1 - x0), out_y0:out_y0 + (y1 - y0), :] = self._volume_at(
            real_t, slice(x0, x1), slice(y0, y1), slice(None)
        )
        return feature

    def _local_flow(self, real_t, x, y):
        radius = self.cnn_nbhd_size
        size = 2 * radius + 1
        feature = np.zeros((size, size, 4), dtype=np.float32)

        for nx in range(max(0, x - radius), min(self.grid_h, x + radius + 1)):
            for ny in range(max(0, y - radius), min(self.grid_w, y + radius + 1)):
                ox = nx - (x - radius)
                oy = ny - (y - radius)
                feature[ox, oy, 0] = self._flow_at(0, real_t, x, y, nx, ny)
                feature[ox, oy, 1] = self._flow_at(0, real_t, nx, ny, x, y)
                feature[ox, oy, 2] = self._flow_at(1, real_t - 1, x, y, nx, ny)
                feature[ox, oy, 3] = self._flow_at(1, real_t - 1, nx, ny, x, y)

        return feature

    def _lstm_feature(self, real_t, x, y):
        hist_feature = self._volume_at(
            slice(real_t - self.hist_feature_daynum * self.timeslot_daynum, real_t, self.timeslot_daynum),
            x,
            y,
            slice(None),
        ).flatten()
        last_feature = self._volume_at(
            slice(real_t - self.last_feature_num, real_t),
            x,
            y,
            slice(None),
        ).flatten()
        nbhd_feature = self._local_volume(real_t, x, y, self.nbhd_size).flatten()
        return np.concatenate((hist_feature, last_feature, nbhd_feature)).astype(np.float32, copy=False)

    def _context_feature(self, real_t):
        if self.context is None:
            return None
        return np.asarray(self.context[real_t], dtype=np.float32)

    def __getitem__(self, idx):
        t, x, y = self._decode_index(idx)

        cnn_features = []
        flow_features = []
        short_term_lstm_samples = []
        context_samples = []
        for seqn in range(self.short_term_lstm_seq_len):
            real_t = t - (self.short_term_lstm_seq_len - seqn)
            cnn_features.append(torch.from_numpy(self._local_volume(real_t, x, y, self.cnn_nbhd_size)).float())
            flow_features.append(torch.from_numpy(self._local_flow(real_t, x, y)).float())
            short_term_lstm_samples.append(self._lstm_feature(real_t, x, y))
            if self.context is not None:
                context_samples.append(self._context_feature(real_t))

        short_term_lstm_features = torch.from_numpy(np.asarray(short_term_lstm_samples, dtype=np.float32)).float()
        if self.context is not None:
            context_features = torch.from_numpy(np.asarray(context_samples, dtype=np.float32)).float()

        att_cnn_features = []
        att_flow_features = []
        att_lstm_features = []
        att_context_features = []
        for att_lstm_cnt in range(self.att_lstm_num):
            long_term_lstm_samples = []
            long_context_samples = []
            att_t = t - (self.att_lstm_num - att_lstm_cnt) * self.timeslot_daynum
            att_t += int((self.long_term_lstm_seq_len - 1) / 2 + 1)

            for seqn in range(self.long_term_lstm_seq_len):
                real_t = att_t - (self.long_term_lstm_seq_len - seqn)
                att_cnn_features.append(torch.from_numpy(self._local_volume(real_t, x, y, self.cnn_nbhd_size)).float())
                att_flow_features.append(torch.from_numpy(self._local_flow(real_t, x, y)).float())
                long_term_lstm_samples.append(self._lstm_feature(real_t, x, y))
                if self.context is not None:
                    long_context_samples.append(self._context_feature(real_t))

            att_lstm_features.append(torch.from_numpy(np.asarray(long_term_lstm_samples, dtype=np.float32)).float())
            if self.context is not None:
                att_context_features.append(torch.from_numpy(np.asarray(long_context_samples, dtype=np.float32)).float())

        # Denormalize label back to original scale for proper loss calculation
        label = torch.from_numpy(self._volume_at(t, x, y, slice(None))).float()

        sample = (
            att_cnn_features,
            att_flow_features,
            att_lstm_features,
            cnn_features,
            flow_features,
            short_term_lstm_features,
            label,
        )
        if self.context is None:
            return sample

        return (*sample, context_features, torch.stack(att_context_features, dim=0))


def create_lazy_dataset(datatype, config_path="data.json", **kwargs):
    return STDNLazyDataset(datatype=datatype, config_path=config_path, **kwargs)
