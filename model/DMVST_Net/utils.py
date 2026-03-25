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


def custom_loss(pred, target, label_max, label_min, mean_label, loss_lambda, eps=1e-5):
    y_true = target * (label_max - label_min) + label_min
    y_pred = pred * (label_max - label_min) + label_min

    diff = (y_true - y_pred) ** 2 / torch.clamp(y_true ** 2, min=eps)

    loss1 = 10.0 * torch.mean(diff)
    loss2 = loss_lambda / (mean_label ** 2 + eps) * torch.mean((y_pred - y_true) ** 2)

    return loss1 + loss2


def get_metrics(y_true, y_pred, max_value, min_value):
    y_true = y_true * (max_value - min_value) + min_value
    y_pred = y_pred * (max_value - min_value) + min_value

    mask = y_true > 10 - 1e-10

    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]

    smape = np.mean(np.abs(2 * (y_true_f - y_pred_f) / (y_true_f + y_pred_f + 1e-5)))
    rmse = np.sqrt(np.mean((y_true_f - y_pred_f) ** 2))
    mape_variant = np.mean(np.abs((y_true_f - y_pred_f) / (y_true_f + 10)))

    return smape, rmse, mape_variant