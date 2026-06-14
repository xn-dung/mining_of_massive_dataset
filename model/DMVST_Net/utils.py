import torch
import numpy as np


def sample_get(datasource, seq_len, cnt):
    X, Y = [], []
    for i in range(datasource.shape[0]):
        if i % cnt < seq_len - 1:
            continue
        tmpx, tmpy = [], []
        for j in range(seq_len):
            tmpx.append(datasource[i - seq_len - 1 + j, :-1])
            if j == seq_len - 1:
                tmpy.append(datasource[i - seq_len - 1 + j, -1])
        X.append(tmpx)
        Y.append(tmpy)
    return np.array(X), np.array(Y)


def sample_get_network(datasource, seq_len, cnt):
    X = []
    for i in range(datasource.shape[0]):
        if i % cnt < seq_len - 1:
            continue
        tmpx = []
        for j in range(seq_len):
            tmpx.append(datasource[i - seq_len - 1 + j, :, :])
        X.append(tmpx)
    return np.array(X)


def sample_get_static(datasource, seq_len, cnt):
    X = []
    for i in range(datasource.shape[0]):
        if i % cnt < seq_len - 1:
            continue
        X.append(datasource[i - seq_len - 1 + (seq_len - 1), :])
    return np.array(X)


def custom_loss(pred, target, label_max, label_min, mean_label=None, loss_lambda=1, eps=1):

    label_max = torch.as_tensor(label_max, dtype=target.dtype, device=target.device)
    label_min = torch.as_tensor(label_min, dtype=target.dtype, device=target.device)

    label_max = label_max.view(1, -1)
    label_min = label_min.view(1, -1)

    y_true = target * (label_max - label_min) + label_min
    y_pred = pred * (label_max - label_min) + label_min
    
    mask = y_true >= 10.0
    
    if mask.any():
        error = y_pred[mask] - y_true[mask]
        pct = torch.mean((error / torch.clamp(y_true[mask], min=eps)) ** 2)
        mse = torch.mean(error ** 2)
    else:
        mse = torch.mean((y_pred - y_true) ** 2)
        pct = (y_pred * 0.0).sum()

    loss = mse + loss_lambda * pct
    return loss


def get_metrics(y_true, y_pred, max_value, min_value, threshold=10.0):

    if isinstance(max_value, torch.Tensor):
        max_value = max_value.cpu().numpy()
    if isinstance(min_value, torch.Tensor):
        min_value = min_value.cpu().numpy()

    max_value = np.array(max_value).reshape(1, -1)
    min_value = np.array(min_value).reshape(1, -1)

    y_true = y_true * (max_value - min_value) + min_value
    y_pred = y_pred * (max_value - min_value) + min_value

    mask = y_true >= threshold

    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]

    if y_true_f.size == 0:
        return np.nan, np.nan, np.nan

    mape = np.mean(np.abs((y_true_f - y_pred_f) / y_true_f))
    rmse = np.sqrt(np.mean((y_true_f - y_pred_f) ** 2))
    mape_variant = np.mean(np.abs((y_true_f - y_pred_f) / (y_true_f + 10)))

    return mape, rmse, mape_variant
