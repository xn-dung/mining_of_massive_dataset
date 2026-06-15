from pathlib import Path

import pandas as pd

from .utils.file_utils import read_table, write_json
from .utils.metrics import compute_regression_metrics


PREDICTION_KEYS = ["time", "grid_x", "grid_y"]


def _normalize_prediction_columns(df):
    rename = {
        "Grid_X": "grid_x",
        "Grid_Y": "grid_y",
        "pred_start_volume": "pred_inflow",
        "pred_end_volume": "pred_outflow",
    }
    df = df.rename(columns={key: value for key, value in rename.items() if key in df.columns})
    return df


def _normalize_actual_columns(df):
    rename = {
        "Grid_X": "grid_x",
        "Grid_Y": "grid_y",
        "start_volume": "actual_inflow",
        "end_volume": "actual_outflow",
        "label_inflow": "actual_inflow",
        "label_outflow": "actual_outflow",
    }
    df = df.rename(columns={key: value for key, value in rename.items() if key in df.columns})
    return df


def _value_pairs(merged):
    actual_cols = []
    pred_cols = []
    if {"actual_inflow", "pred_inflow"}.issubset(merged.columns):
        actual_cols.append("actual_inflow")
        pred_cols.append("pred_inflow")
    if {"actual_outflow", "pred_outflow"}.issubset(merged.columns):
        actual_cols.append("actual_outflow")
        pred_cols.append("pred_outflow")

    if not actual_cols:
        raise ValueError(
            "Could not find matching actual/prediction value columns. "
            "Expected actual_inflow/pred_inflow and optionally actual_outflow/pred_outflow."
        )
    return merged[actual_cols].to_numpy().reshape(-1), merged[pred_cols].to_numpy().reshape(-1)


def evaluate_prediction(prediction_path, actual_path):
    pred = _normalize_prediction_columns(read_table(prediction_path))
    actual = _normalize_actual_columns(read_table(actual_path))

    missing_pred = [col for col in PREDICTION_KEYS if col not in pred.columns]
    missing_actual = [col for col in PREDICTION_KEYS if col not in actual.columns]
    if missing_pred or missing_actual:
        raise ValueError(f"Missing merge keys. prediction={missing_pred}, actual={missing_actual}")

    pred["time"] = pd.to_datetime(pred["time"]).astype(str)
    actual["time"] = pd.to_datetime(actual["time"]).astype(str)
    merged = pred.merge(actual, on=PREDICTION_KEYS, how="inner")
    if merged.empty:
        raise ValueError("Prediction and actual files did not overlap on time/grid_x/grid_y")

    actual_values, pred_values = _value_pairs(merged)
    metrics = compute_regression_metrics(actual_values, pred_values)
    metrics["n_rows"] = int(len(merged))
    return metrics


def evaluate_prediction_frame(prediction_path, start_date=None, end_date=None):
    pred = _normalize_prediction_columns(read_table(prediction_path))
    missing = [col for col in PREDICTION_KEYS if col not in pred.columns]
    if missing:
        raise ValueError(f"Missing prediction keys: {missing}")

    pred["time"] = pd.to_datetime(pred["time"])
    if start_date is not None:
        pred = pred[pred["time"].dt.date >= pd.to_datetime(start_date).date()]
    if end_date is not None:
        pred = pred[pred["time"].dt.date <= pd.to_datetime(end_date).date()]
    if pred.empty:
        raise ValueError(f"No prediction rows in evaluation window {start_date} -> {end_date}")

    actual_values, pred_values = _value_pairs(pred)
    metrics = compute_regression_metrics(actual_values, pred_values)
    metrics["n_rows"] = int(len(pred))
    return metrics


def save_metrics(metrics, output_path):
    output_path = Path(output_path)
    write_json(output_path, metrics)
    return output_path
