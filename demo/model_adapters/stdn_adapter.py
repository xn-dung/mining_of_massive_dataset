import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from .. import config
from ..evaluate import evaluate_prediction
from ..notifier import notify
from ..utils.file_utils import ensure_dir
from ..utils.zone_lookup import enrich_prediction_frame


def _ensure_training_path():
    training_dir = str(config.TRAINING_DIR)
    if training_dir not in sys.path:
        sys.path.insert(0, training_dir)


def _load_training_modules():
    _ensure_training_path()
    import train_config as base_cfg
    import train_context_stdn
    import train_utils

    return base_cfg, train_context_stdn, train_utils


def _make_runtime_config(data_root, model_path, output_root=None):
    base_cfg, _, _ = _load_training_modules()
    attrs = {
        name: getattr(base_cfg, name)
        for name in dir(base_cfg)
        if name.isupper()
    }
    data_root = Path(data_root)
    output_root = Path(output_root or config.ARTIFACT_DIR / "stdn_runtime")
    ensure_dir(output_root)

    attrs.update(
        {
            "PROJECT_ROOT": config.PROJECT_ROOT,
            "STDN_DIR": config.STDN_DIR,
            "TRAINING_DIR": config.TRAINING_DIR,
            "DATA_ROOT": data_root,
            "EVAL_DATA_ROOT": data_root,
            "SOURCE_DATA_CONFIG": data_root / "data.json",
            "RUNTIME_DATA_CONFIG": output_root / "runtime_data.json",
            "MODEL_SAVE_PATH": Path(model_path),
            "TRAIN_TEST_PREDICTIONS_PATH": output_root / "train_test_predictions.npz",
            "TRAIN_TEST_METRICS_PATH": output_root / "train_test_metrics.json",
            "TRAIN_TEST_EXPERIMENT_PATH": output_root / "train_test_experiment.json",
            "TRAIN_TEST_PLOT_PATH": output_root / "train_test_48h_plot.png",
            "EVAL_PREDICTIONS_PATH": output_root / "eval_predictions.npz",
            "EVAL_METRICS_PATH": output_root / "eval_metrics.json",
            "EVAL_EXPERIMENT_PATH": output_root / "eval_experiment.json",
            "EVAL_PLOT_PATH": output_root / "eval_48h_plot.png",
            "MAX_EPOCHS": config.DEMO_TRAIN_EPOCHS,
            "OUTPUT_SHAPE": config.OUTPUT_SHAPE,
            "USE_TQDM": True,
        }
    )
    return SimpleNamespace(**attrs)


def _checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        return checkpoint.get("model_state_dict", checkpoint)
    return checkpoint


def _load_checkpoint_into_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(_checkpoint_state_dict(checkpoint))
    return checkpoint


def _prediction_frame(metrics, shape_info, data_config, model_version=None):
    preds = np.asarray(metrics["preds_raw"])
    targets = np.asarray(metrics["targets_raw"])
    cell_count = int(shape_info["grid_h"]) * int(shape_info["grid_w"])
    n_time = min(len(preds), len(targets)) // cell_count
    usable = n_time * cell_count
    preds = preds[:usable]
    targets = targets[:usable]

    volume_path = Path(data_config.get("volume_test") or data_config.get("volume"))
    times = None
    if volume_path.exists():
        volume_npz = np.load(volume_path, allow_pickle=True)
        if "time" in volume_npz.files:
            all_times = pd.to_datetime(volume_npz["time"]).astype(str)
            start = int(shape_info.get("time_start", 0))
            times = all_times[start:start + n_time]
    if times is None or len(times) < n_time:
        times = [str(i) for i in range(n_time)]

    grid_h = int(shape_info["grid_h"])
    grid_w = int(shape_info["grid_w"])
    rows = []
    for time_idx in range(n_time):
        for cell in range(cell_count):
            sample_idx = time_idx * cell_count + cell
            x = cell // grid_w
            y = cell % grid_w
            row = {
                "time": str(times[time_idx]),
                "grid_x": x,
                "grid_y": y,
                "pred_inflow": float(preds[sample_idx, 0]),
                "actual_inflow": float(targets[sample_idx, 0]),
                "predicted_inflow_trips": float(preds[sample_idx, 0]),
                "actual_inflow_trips": float(targets[sample_idx, 0]),
                "model_version": model_version,
            }
            if preds.shape[1] > 1:
                row["pred_outflow"] = float(preds[sample_idx, 1])
                row["predicted_outflow_trips"] = float(preds[sample_idx, 1])
            if targets.shape[1] > 1:
                row["actual_outflow"] = float(targets[sample_idx, 1])
                row["actual_outflow_trips"] = float(targets[sample_idx, 1])
            rows.append(row)
    return pd.DataFrame(rows)


def load_model(model_path, processed_data_path=None):
    _, _, train_utils = _load_training_modules()
    if processed_data_path is None:
        raise ValueError("processed_data_path is required. The demo does not fall back to external datasets.")
    data_root = Path(processed_data_path)
    runtime_cfg = _make_runtime_config(data_root, model_path)
    test_loader, shape_info, data_config, _ = train_utils.create_test_loader_from_data_root(runtime_cfg, data_root)
    model, device = train_utils.create_model(runtime_cfg, shape_info, data_config)
    checkpoint = _load_checkpoint_into_model(model, model_path, device)
    return {
        "model": model,
        "device": device,
        "checkpoint": checkpoint,
        "shape_info": shape_info,
        "data_config": data_config,
        "test_loader": test_loader,
        "runtime_config": runtime_cfg,
    }


def predict_period(model_path, processed_data_path, output_path, model_version=None):
    _, _, train_utils = _load_training_modules()
    data_root = Path(processed_data_path)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    print(f"[stdn adapter] Loading processed data: {data_root}", flush=True)
    runtime_cfg = _make_runtime_config(data_root, model_path, output_root=output_path.parent / "_runtime")
    test_loader, shape_info, data_config, _ = train_utils.create_test_loader_from_data_root(runtime_cfg, data_root)
    print(f"[stdn adapter] Loading model: {model_path}", flush=True)
    model, device = train_utils.create_model(runtime_cfg, shape_info, data_config)
    _load_checkpoint_into_model(model, model_path, device)

    print(f"[stdn adapter] Running STDN prediction/evaluation on {len(test_loader)} batches", flush=True)
    metrics = train_utils.evaluate(model, test_loader, data_config, device, runtime_cfg, desc="STDN predict")
    prediction_df = _prediction_frame(metrics, shape_info, data_config, model_version=model_version)
    prediction_df = enrich_prediction_frame(prediction_df)
    prediction_df.to_csv(output_path, index=False)
    print(f"[stdn adapter] Saved prediction: {output_path}", flush=True)
    return {
        "prediction_path": output_path,
        "metrics": {
            "wmape": metrics.get("wmape"),
            "mape": metrics.get("mape"),
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
        },
        "shape_info": shape_info,
        "data_config": data_config,
    }


def evaluate_model(model_path, processed_data_path, actual_path=None, model_version=None):
    period_name = Path(processed_data_path).name
    prediction_path = config.PREDICTION_DIR / f"{period_name}_{model_version or 'model'}_prediction.csv"
    result = predict_period(model_path, processed_data_path, prediction_path, model_version=model_version)

    if actual_path:
        metrics = evaluate_prediction(prediction_path, actual_path)
    else:
        metrics = result["metrics"]
    return {
        "prediction_path": prediction_path,
        "metrics": metrics,
        "shape_info": result["shape_info"],
        "data_config": result["data_config"],
    }


def train_model(train_data_path, base_checkpoint=None, model_version=None, output_model_path=None):
    _, train_context_stdn, _ = _load_training_modules()
    train_data_path = Path(train_data_path)
    output_model_path = Path(output_model_path or config.MODEL_DIR / f"{model_version or 'stdn_candidate'}.pth")
    ensure_dir(output_model_path.parent)

    if base_checkpoint and Path(base_checkpoint).exists() and not output_model_path.exists():
        shutil.copy2(base_checkpoint, output_model_path)

    data_config_path = train_data_path / "data.json"
    if not data_config_path.exists():
        raise FileNotFoundError(f"Missing processed train config: {data_config_path}")

    runtime_cfg = _make_runtime_config(
        train_data_path,
        output_model_path,
        output_root=config.ARTIFACT_DIR / "training_runs" / (model_version or "candidate"),
    )
    train_context_stdn.run(runtime_cfg)
    return output_model_path
