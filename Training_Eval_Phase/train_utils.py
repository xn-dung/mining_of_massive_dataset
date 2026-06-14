import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ARRAY_KEYS = (
    "volume_train",
    "volume_test",
    "flow_train",
    "flow_test",
    "context_train",
    "context_test",
)


def add_stdn_to_path(stdn_dir):
    stdn_dir = str(Path(stdn_dir))
    if stdn_dir not in sys.path:
        sys.path.insert(0, stdn_dir)


def load_data_config(cfg, data_root=None, source_data_config=None, runtime_data_config=None):
    """Load data.json and rewrite array paths to the configured DATA_ROOT."""
    data_root = Path(data_root) if data_root is not None else Path(cfg.DATA_ROOT)
    if source_data_config is None and data_root != Path(cfg.DATA_ROOT):
        source_path = data_root / "data.json"
    else:
        source_path = Path(source_data_config or cfg.SOURCE_DATA_CONFIG)

    if not source_path.exists():
        raise FileNotFoundError(f"Missing dataset config: {source_path}")

    with source_path.open("r", encoding="utf-8") as f:
        data_config = json.load(f)

    for key in ARRAY_KEYS:
        if key in data_config:
            data_config[key] = str(data_root / Path(data_config[key]).name)

    runtime_path = Path(runtime_data_config or cfg.RUNTIME_DATA_CONFIG)
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    with runtime_path.open("w", encoding="utf-8") as f:
        json.dump(data_config, f, indent=2)

    return data_config, runtime_path


def load_stdn_objects(cfg):
    add_stdn_to_path(cfg.STDN_DIR)
    from file_loader_lazy import STDNLazyDataset, collate_fn
    from models import models as model_factory

    return STDNLazyDataset, collate_fn, model_factory


def create_train_val_loaders(cfg):
    STDNLazyDataset, collate_fn, _ = load_stdn_objects(cfg)
    data_config, runtime_config = load_data_config(cfg)

    dataset = STDNLazyDataset(
        "train",
        config_path=str(runtime_config),
        att_lstm_num=cfg.ATT_LSTM_NUM,
        long_term_lstm_seq_len=cfg.LONG_TERM_LSTM_SEQ_LEN,
        short_term_lstm_seq_len=cfg.SHORT_TERM_LSTM_SEQ_LEN,
        hist_feature_daynum=cfg.HIST_FEATURE_DAYNUM,
        nbhd_size=cfg.NBHD_SIZE,
        cnn_nbhd_size=cfg.CNN_NBHD_SIZE,
        cache_dir=cfg.DATA_ROOT,
    )

    n_val_time = int(dataset.target_time_count * cfg.VALIDATION_SPLIT)
    n_train_time = dataset.target_time_count - n_val_time
    train_stop = n_train_time * dataset.cell_count

    train_indices = range(0, train_stop)
    val_indices = range(train_stop, len(dataset))

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    return train_loader, val_loader, dataset.sample_shape_info(), data_config, runtime_config


def create_test_loader(cfg, runtime_config=None, cache_dir=None):
    STDNLazyDataset, collate_fn, _ = load_stdn_objects(cfg)
    if runtime_config is None:
        _, runtime_config = load_data_config(cfg)

    dataset = STDNLazyDataset(
        "test",
        config_path=str(runtime_config),
        att_lstm_num=cfg.ATT_LSTM_NUM,
        long_term_lstm_seq_len=cfg.LONG_TERM_LSTM_SEQ_LEN,
        short_term_lstm_seq_len=cfg.SHORT_TERM_LSTM_SEQ_LEN,
        hist_feature_daynum=cfg.HIST_FEATURE_DAYNUM,
        nbhd_size=cfg.NBHD_SIZE,
        cnn_nbhd_size=cfg.CNN_NBHD_SIZE,
        cache_dir=cache_dir or cfg.DATA_ROOT,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    return loader, dataset.sample_shape_info()


def create_test_loader_from_data_root(cfg, data_root):
    data_config, runtime_config = load_data_config(cfg, data_root=data_root)
    loader, shape_info = create_test_loader(cfg, runtime_config=runtime_config, cache_dir=data_root)
    return loader, shape_info, data_config, runtime_config


def create_model(cfg, shape_info, data_config, device=None):
    _, _, model_factory = load_stdn_objects(cfg)
    context_dim = int(shape_info.get("context_dim") or data_config.get("context_dim", 0))
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_factory().stdn(
        att_lstm_num=cfg.ATT_LSTM_NUM,
        att_lstm_seq_len=cfg.LONG_TERM_LSTM_SEQ_LEN,
        lstm_seq_len=cfg.SHORT_TERM_LSTM_SEQ_LEN,
        feature_vec_len=shape_info["feature_vec_len"],
        cnn_flat_size=cfg.CNN_FLAT_SIZE,
        lstm_out_size=cfg.LSTM_OUT_SIZE,
        nbhd_size=shape_info["cnn_nbhd_dim"],
        nbhd_type=shape_info["volume_type"],
        flow_type=4,
        output_shape=cfg.OUTPUT_SHAPE,
        dropout_rate=cfg.DROPOUT_RATE,
        context_dim=context_dim,
    ).to(device)
    return model, device


def _to_device_list(items, device):
    return [item.to(device, non_blocking=True) for item in items]


def move_batch_to_device(batch, device):
    has_context = len(batch) == 9
    att_cnn, att_flow, att_lstm, cnn, flow, lstm, labels = batch[:7]

    moved = {
        "att_cnn": _to_device_list(att_cnn, device),
        "att_flow": _to_device_list(att_flow, device),
        "att_lstm": _to_device_list(att_lstm, device),
        "cnn": _to_device_list(cnn, device),
        "flow": _to_device_list(flow, device),
        "lstm": lstm.to(device, non_blocking=True),
        "labels": labels.to(device, non_blocking=True),
        "context": None,
        "att_context": None,
    }

    if has_context:
        moved["context"] = batch[7].to(device, non_blocking=True)
        moved["att_context"] = batch[8].to(device, non_blocking=True)

    return moved


def forward_batch(model, batch):
    return model(
        batch["att_cnn"],
        batch["att_flow"],
        batch["att_lstm"],
        batch["cnn"],
        batch["flow"],
        batch["lstm"],
        context_inputs=batch["context"],
        att_context_inputs=batch["att_context"],
    )


def train_one_epoch(model, loader, criterion, optimizer, device, cfg, epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, raw_batch in enumerate(loader, start=1):
        batch = move_batch_to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = forward_batch(model, batch)
        loss = criterion(outputs, batch["labels"])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % cfg.PRINT_EVERY == 0:
            print(f"Epoch {epoch} Batch [{batch_idx}/{len(loader)}] - Loss: {loss.item():.6f}")

    return running_loss / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        outputs = forward_batch(model, batch)
        total_loss += criterion(outputs, batch["labels"]).item()
    return total_loss / max(1, len(loader))


def compute_metrics(preds_norm, targets_norm, data_config, cfg):
    raw_max = float(data_config.get("raw_volume_train_max", data_config.get("volume_train_max", 1.0))) + 1e-6
    preds_raw = preds_norm * raw_max
    targets_raw = targets_norm * raw_max

    err = preds_raw - targets_raw
    rmse = float(math.sqrt(np.mean(err ** 2)))
    wmape = float(np.sum(np.abs(err)) / (np.sum(np.abs(targets_raw)) + cfg.MAPE_EPS) * 100.0)

    filtered_mask = targets_raw > float(getattr(cfg, "FILTERED_MAPE_THRESHOLD", 10.0))
    if np.any(filtered_mask):
        filtered_mape = float(np.mean(np.abs(err[filtered_mask]) / targets_raw[filtered_mask]) * 100.0)
    else:
        filtered_mape = float("nan")

    target_flat = targets_raw.reshape(-1)
    pred_flat = preds_raw.reshape(-1)
    ss_res = float(np.sum((target_flat - pred_flat) ** 2))
    ss_tot = float(np.sum((target_flat - np.mean(target_flat)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + cfg.MAPE_EPS))

    return {
        "rmse": rmse,
        "wmape": wmape,
        "filtered_mape": filtered_mape,
        "r2": r2,
        "preds_raw": preds_raw,
        "targets_raw": targets_raw,
        "raw_max": raw_max,
    }


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    preds = []
    targets = []
    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        outputs = forward_batch(model, batch)
        preds.append(outputs.detach().cpu().numpy())
        targets.append(batch["labels"].detach().cpu().numpy())

    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


@torch.no_grad()
def validate_with_metrics(model, loader, criterion, data_config, device, cfg):
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []
    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        outputs = forward_batch(model, batch)
        total_loss += criterion(outputs, batch["labels"]).item()
        preds.append(outputs.detach().cpu().numpy())
        targets.append(batch["labels"].detach().cpu().numpy())

    preds_norm = np.concatenate(preds, axis=0)
    targets_norm = np.concatenate(targets, axis=0)
    metrics = compute_metrics(preds_norm, targets_norm, data_config, cfg)
    metrics["loss"] = total_loss / max(1, len(loader))
    return metrics


@torch.no_grad()
def evaluate(model, loader, data_config, device, cfg):
    preds_norm, targets_norm = collect_predictions(model, loader, device)
    metrics = compute_metrics(preds_norm, targets_norm, data_config, cfg)
    metrics["mae"] = float(np.mean(np.abs(metrics["preds_raw"] - metrics["targets_raw"])))
    metrics["mape"] = metrics["filtered_mape"]
    return metrics


def save_prediction_outputs(metrics, predictions_path, metrics_path):
    predictions_path = Path(predictions_path)
    metrics_path = Path(metrics_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        predictions_path,
        preds_raw=metrics["preds_raw"],
        targets_raw=metrics["targets_raw"],
        raw_max=np.asarray(metrics["raw_max"], dtype=np.float32),
    )

    payload = {
        "rmse": metrics["rmse"],
        "wmape": metrics["wmape"],
        "filtered_mape": metrics["filtered_mape"],
        "r2": metrics["r2"],
        "raw_max": metrics["raw_max"],
        "predictions_path": str(predictions_path),
    }
    if "loss" in metrics:
        payload["loss"] = metrics["loss"]
    if "mae" in metrics:
        payload["mae"] = metrics["mae"]

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return predictions_path, metrics_path


def save_checkpoint(model, cfg, extra=None):
    save_path = Path(cfg.MODEL_SAVE_PATH)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, save_path)
    return save_path
