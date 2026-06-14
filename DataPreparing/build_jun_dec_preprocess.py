import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import build_jun_dec_config as cfg


def log(message):
    print(f"[build_jun_dec] {message}", flush=True)


def month_paths(month):
    mm = f"{month:02d}"
    direct_volume = cfg.RAW_BASE_DIR / f"{mm}_volume.csv"
    direct_flow = cfg.RAW_BASE_DIR / f"{mm}_flow.parquet"
    if direct_volume.exists() or direct_flow.exists():
        return direct_volume, direct_flow

    year_dir = cfg.RAW_BASE_DIR / f"{cfg.DATASET}_data" / f"{cfg.DATASET}_{cfg.YEAR}"
    return year_dir / f"{mm}_volume.csv", year_dir / f"{mm}_flow.parquet"


def build_time_index():
    return pd.date_range(pd.Timestamp(cfg.START_TIME), pd.Timestamp(cfg.END_TIME), freq=cfg.TIME_INTERVAL)


def build_grid_mapping(lookup_path):
    grid = pd.read_csv(lookup_path)
    grid["Grid_X"] = grid["Grid_X"].astype(int)
    grid["Grid_Y"] = grid["Grid_Y"].astype(int)
    return dict(zip(grid["LocationID"], zip(grid["Grid_X"], grid["Grid_Y"])))


def load_location_mapping():
    lookup_path = Path(cfg.LOOKUP_PATH)
    if not lookup_path.exists():
        raise FileNotFoundError(f"Lookup file not found: {lookup_path}")

    log(f"Loading lookup grid: {lookup_path}")
    loc2ij = build_grid_mapping(lookup_path)
    log(f"Loaded {len(loc2ij)} location-grid mappings")
    return loc2ij


def validate_input_files():
    missing = []
    if not cfg.RAW_BASE_DIR.exists():
        missing.append(str(cfg.RAW_BASE_DIR))
    if not cfg.LOOKUP_PATH.exists():
        missing.append(str(cfg.LOOKUP_PATH))
    if not cfg.CONTEXT_PATH.exists():
        missing.append(str(cfg.CONTEXT_PATH))

    for month in cfg.MONTHS:
        vol_path, flow_path = month_paths(month)
        if not vol_path.exists():
            missing.append(str(vol_path))
        if not flow_path.exists():
            missing.append(str(flow_path))

    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def load_context_for_times(time_index):
    if not cfg.CONTEXT_PATH.exists():
        raise FileNotFoundError(f"Context file not found: {cfg.CONTEXT_PATH}")

    log(f"Loading context: {cfg.CONTEXT_PATH}")
    context_npz = np.load(cfg.CONTEXT_PATH, allow_pickle=True)
    context = context_npz["context"].astype(np.float32)
    context_time = pd.to_datetime(context_npz["time"])
    feature_names = context_npz["feature_names"].astype(str)

    context_df = pd.DataFrame(context, index=context_time, columns=feature_names)
    aligned = context_df.reindex(time_index)

    missing = aligned[aligned.isna().any(axis=1)].index
    if len(missing) > 0:
        raise ValueError(
            "Context is missing expected time bins. First missing bins: "
            + ", ".join(str(t) for t in missing[:10])
        )

    aligned_context = aligned.to_numpy(dtype=np.float32)
    log(
        "Context aligned: "
        f"shape={aligned_context.shape}, "
        f"time={time_index[0]} -> {time_index[-1]}, "
        f"range=({aligned_context.min():.6f}, {aligned_context.max():.6f})"
    )
    log("Context features: " + ", ".join(feature_names.tolist()))

    return aligned_context, feature_names


def process_volume_month(vol_path, volume, time_map, loc2ij):
    if not vol_path.exists():
        log(f"Skip volume, file not found: {vol_path}")
        return 0, 0

    vol_df = pd.read_csv(vol_path)
    raw_rows = len(vol_df)
    vol_df["slot"] = pd.to_datetime(vol_df["time_bin"], errors="coerce").dt.floor(cfg.TIME_INTERVAL)

    location_col = "locationid" if "locationid" in vol_df.columns else "LocationID"
    vol_df["LocationID"] = pd.to_numeric(vol_df[location_col], errors="coerce")
    vol_df = vol_df.dropna(subset=["slot", "LocationID"])
    vol_df["LocationID"] = vol_df["LocationID"].astype(int)

    vol_hour = vol_df.groupby(["slot", "LocationID"], as_index=False)[["start_volume", "end_volume"]].sum()
    valid_mask = vol_hour["LocationID"].isin(loc2ij) & vol_hour["slot"].isin(time_map)
    vol_valid = vol_hour[valid_mask].copy()
    if vol_valid.empty:
        return raw_rows, 0

    coords = vol_valid["LocationID"].map(loc2ij)
    t_idx = vol_valid["slot"].map(time_map).to_numpy()
    i_idx = np.array([c[0] for c in coords], dtype=np.int64)
    j_idx = np.array([c[1] for c in coords], dtype=np.int64)

    np.add.at(volume[..., 0], (t_idx, i_idx, j_idx), vol_valid["start_volume"].to_numpy(dtype=np.float32))
    np.add.at(volume[..., 1], (t_idx, i_idx, j_idx), vol_valid["end_volume"].to_numpy(dtype=np.float32))
    return raw_rows, len(vol_valid)


def process_flow_month(flow_path, flow, time_map, loc2ij):
    if not flow_path.exists():
        log(f"Skip flow, file not found: {flow_path}")
        return 0, 0

    pf = pq.ParquetFile(flow_path)
    flow_df = pf.read(columns=["time_bin", "pulocationid", "dolocationid", "flow_count"]).to_pandas()
    raw_rows = len(flow_df)
    flow_df["slot"] = pd.to_datetime(flow_df["time_bin"], errors="coerce").dt.floor(cfg.TIME_INTERVAL)
    flow_df = flow_df.dropna(subset=["slot", "pulocationid", "dolocationid"])

    flow_hour = flow_df.groupby(["slot", "pulocationid", "dolocationid"], as_index=False)["flow_count"].sum()
    valid_mask = (
        flow_hour["pulocationid"].isin(loc2ij)
        & flow_hour["dolocationid"].isin(loc2ij)
        & flow_hour["slot"].isin(time_map)
    )
    flow_valid = flow_hour[valid_mask].copy()
    if flow_valid.empty:
        return raw_rows, 0

    pu_coords = flow_valid["pulocationid"].map(loc2ij)
    do_coords = flow_valid["dolocationid"].map(loc2ij)

    t_idx = flow_valid["slot"].map(time_map).to_numpy()
    i_idx = np.array([c[0] for c in pu_coords], dtype=np.int64)
    j_idx = np.array([c[1] for c in pu_coords], dtype=np.int64)
    k_idx = np.array([c[0] for c in do_coords], dtype=np.int64)
    l_idx = np.array([c[1] for c in do_coords], dtype=np.int64)
    counts = flow_valid["flow_count"].to_numpy(dtype=np.float32)

    np.add.at(flow[0], (t_idx, i_idx, j_idx, k_idx, l_idx), counts)

    valid_prev = t_idx > 0
    np.add.at(
        flow[1],
        (
            t_idx[valid_prev] - 1,
            i_idx[valid_prev],
            j_idx[valid_prev],
            k_idx[valid_prev],
            l_idx[valid_prev],
        ),
        counts[valid_prev],
    )
    return raw_rows, len(flow_valid)


def save_split(volume, flow, context, time_index, feature_names):
    if getattr(cfg, "TEST_ONLY_FULL", False):
        save_full_test(volume, flow, context, time_index, feature_names)
        return

    split = int(len(time_index) * cfg.TRAIN_SPLIT)
    log(f"Splitting by time: split_index={split}, train_slots={split}, test_slots={len(time_index) - split}")

    time_str = time_index.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
    time_train = time_str[:split]
    time_test = time_str[split:]

    volume_train = np.asarray(volume[:split])
    volume_test = np.asarray(volume[split:])
    flow_train = np.asarray(flow[:, :split])
    flow_test = np.asarray(flow[:, split:])
    context_train = context[:split]
    context_test = context[split:]

    log(f"Train time range: {time_train[0]} -> {time_train[-1]}")
    log(f"Test time range: {time_test[0]} -> {time_test[-1]}")

    volume_train_max = float(volume_train.max())
    flow_train_max = float(flow_train.max())
    log(f"Raw max from train only: volume={volume_train_max}, flow={flow_train_max}")

    volume_train_norm = volume_train / (volume_train_max + 1e-6)
    volume_test_norm = volume_test / (volume_train_max + 1e-6)
    flow_train_norm = flow_train / (flow_train_max + 1e-6)
    flow_test_norm = flow_test / (flow_train_max + 1e-6)

    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.OUT_DIR / "volume_train.npz", volume=volume_train_norm, time=time_train)
    log("Saved volume_train.npz")
    np.savez_compressed(cfg.OUT_DIR / "volume_test.npz", volume=volume_test_norm, time=time_test)
    log("Saved volume_test.npz")
    np.savez_compressed(cfg.OUT_DIR / "flow_train.npz", flow=flow_train_norm, time=time_train)
    log("Saved flow_train.npz")
    np.savez_compressed(cfg.OUT_DIR / "flow_test.npz", flow=flow_test_norm, time=time_test)
    log("Saved flow_test.npz")
    np.savez_compressed(cfg.OUT_DIR / "context_train.npz", context=context_train, time=time_train, feature_names=feature_names)
    log("Saved context_train.npz")
    np.savez_compressed(cfg.OUT_DIR / "context_test.npz", context=context_test, time=time_test, feature_names=feature_names)
    log("Saved context_test.npz")

    data_config = {
        "volume_train": str(cfg.OUT_DIR / "volume_train.npz"),
        "volume_test": str(cfg.OUT_DIR / "volume_test.npz"),
        "flow_train": str(cfg.OUT_DIR / "flow_train.npz"),
        "flow_test": str(cfg.OUT_DIR / "flow_test.npz"),
        "context_train": str(cfg.OUT_DIR / "context_train.npz"),
        "context_test": str(cfg.OUT_DIR / "context_test.npz"),
        "volume_train_max": 1.0,
        "flow_train_max": 1.0,
        "raw_volume_train_max": volume_train_max,
        "raw_flow_train_max": flow_train_max,
        "timeslot_sec": cfg.TIMESLOT_SEC,
        "threshold": cfg.THRESHOLD,
        "context_dim": int(context.shape[1]),
        "context_feature_names": feature_names.tolist(),
        "start_time": time_str[0],
        "end_time": time_str[-1],
        "train_split": cfg.TRAIN_SPLIT,
    }
    with (cfg.OUT_DIR / "data.json").open("w", encoding="utf-8") as f:
        json.dump(data_config, f, indent=2)
    log("Saved data.json")

    log(f"Saved dataset to: {cfg.OUT_DIR}")
    log(f"Volume train/test: {volume_train_norm.shape} / {volume_test_norm.shape}")
    log(f"Flow train/test: {flow_train_norm.shape} / {flow_test_norm.shape}")
    log(f"Context train/test: {context_train.shape} / {context_test.shape}")
    log("Time alignment check passed inside each split: volume_time == flow_time == context_time")


def save_full_test(volume, flow, context, time_index, feature_names):
    log("TEST_ONLY_FULL=True, saving the full time range as the test split")

    time_str = time_index.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
    volume_full = np.asarray(volume)
    flow_full = np.asarray(flow)

    log(f"Test-only time range: {time_str[0]} -> {time_str[-1]}")

    volume_max = float(volume_full.max())
    flow_max = float(flow_full.max())
    log(f"Raw max from full test range: volume={volume_max}, flow={flow_max}")

    volume_norm = volume_full / (volume_max + 1e-6)
    flow_norm = flow_full / (flow_max + 1e-6)

    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.OUT_DIR / "volume_test.npz", volume=volume_norm, time=time_str)
    log("Saved volume_test.npz")
    np.savez_compressed(cfg.OUT_DIR / "flow_test.npz", flow=flow_norm, time=time_str)
    log("Saved flow_test.npz")
    np.savez_compressed(cfg.OUT_DIR / "context_test.npz", context=context, time=time_str, feature_names=feature_names)
    log("Saved context_test.npz")

    data_config = {
        "volume_train": str(cfg.OUT_DIR / "volume_test.npz"),
        "volume_test": str(cfg.OUT_DIR / "volume_test.npz"),
        "flow_train": str(cfg.OUT_DIR / "flow_test.npz"),
        "flow_test": str(cfg.OUT_DIR / "flow_test.npz"),
        "context_train": str(cfg.OUT_DIR / "context_test.npz"),
        "context_test": str(cfg.OUT_DIR / "context_test.npz"),
        "volume_train_max": 1.0,
        "flow_train_max": 1.0,
        "raw_volume_train_max": volume_max,
        "raw_flow_train_max": flow_max,
        "timeslot_sec": cfg.TIMESLOT_SEC,
        "threshold": cfg.THRESHOLD,
        "context_dim": int(context.shape[1]),
        "context_feature_names": feature_names.tolist(),
        "start_time": time_str[0],
        "end_time": time_str[-1],
        "train_split": None,
        "test_only_full": True,
    }
    with (cfg.OUT_DIR / "data.json").open("w", encoding="utf-8") as f:
        json.dump(data_config, f, indent=2)
    log("Saved data.json")

    log(f"Saved full-test dataset to: {cfg.OUT_DIR}")
    log(f"Volume test: {volume_norm.shape}")
    log(f"Flow test: {flow_norm.shape}")
    log(f"Context test: {context.shape}")
    log("Time alignment check passed: volume_time == flow_time == context_time")
