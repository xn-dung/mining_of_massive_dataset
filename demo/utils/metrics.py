import numpy as np


EPS = 1e-6


def compute_mae(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def compute_rmse(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def compute_mape(actual, predicted, eps=EPS):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.abs(actual) > eps
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / np.abs(actual[mask])) * 100.0)


def compute_wmape(actual, predicted, eps=EPS):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sum(np.abs(actual - predicted)) / (np.sum(np.abs(actual)) + eps) * 100.0)


def compute_regression_metrics(actual, predicted):
    return {
        "wmape": compute_wmape(actual, predicted),
        "mape": compute_mape(actual, predicted),
        "rmse": compute_rmse(actual, predicted),
        "mae": compute_mae(actual, predicted),
    }
