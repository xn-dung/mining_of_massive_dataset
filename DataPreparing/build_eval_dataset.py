import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import build_eval_config as cfg


def log(message):
    print(f"[build_eval_dataset] {message}", flush=True)


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


def build_grid_mapping():
    grid = pd.read_csv(cfg.LOOKUP_PATH)
    grid["Grid_X"] = grid["Grid_X"].astype(int)
    grid["Grid_Y"] = grid["Grid_Y"].astype(int)
    return dict(zip(grid["LocationID"], zip(grid["Grid_X"], grid["Grid_Y"])))


def validate_input_files():
    missing = []
    for path in (cfg.RAW_BASE_DIR, cfg.LOOKUP_PATH, cfg.CONTEXT_PATH):
        if not path.exists():
            missing.append(str(path))

    for month in cfg.MONTHS:
        volume_path, flow_path = month_paths(month)
        if not volume_path.exists():
            missing.append(str(volume_path))
        if not flow_path.exists():
            missing.append(str(flow_path))

    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def load_context_for_times(time_index):
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
        f"time={time_index[0]} -> {time_index[-1]}"
    )
    return aligned_context, feature_names


def process_volume_month(volume_path, volume, time_map, loc2ij):
    vol_df = pd.read_csv(volume_path)
    raw_rows = len(vol_df)
    vol_df["slot"] = pd.to_datetime(vol_df["time_bin"], errors="coerce").dt.floor(cfg.TIME_INTERVAL)

    location_col = "locationid" if "locationid" in vol_df.columns else "LocationID"
    vol_df["LocationID"] = pd.to_numeric(vol_df[location_col], errors="coerce")
    vol_df = vol_df.dropna(subset=["slot", "LocationID"])
    vol_df["LocationID"] = vol_df["LocationID"].astype(int)

    vol_slot = vol_df.groupby(["slot", "LocationID"], as_index=False)[["start_volume", "end_volume"]].sum()
    valid = vol_slot["LocationID"].isin(loc2ij) & vol_slot["slot"].isin(time_map)
    vol_valid = vol_slot[valid].copy()
    if vol_valid.empty:
        return raw_rows, 0

    coords = vol_valid["LocationID"].map(loc2ij)
    t_idx = vol_valid["slot"].map(time_map).to_numpy()
    i_idx = np.array([coord[0] for coord in coords], dtype=np.int64)
    j_idx = np.array([coord[1] for coord in coords], dtype=np.int64)

    np.add.at(volume[..., 0], (t_idx, i_idx, j_idx), vol_valid["start_volume"].to_numpy(dtype=np.float32))
    np.add.at(volume[..., 1], (t_idx, i_idx, j_idx), vol_valid["end_volume"].to_numpy(dtype=np.float32))
    return raw_rows, len(vol_valid)


def process_flow_month(flow_path, flow, time_map, loc2ij):
    pf = pq.ParquetFile(flow_path)
    flow_df = pf.read(columns=["time_bin", "pulocationid", "dolocationid", "flow_count"]).to_pandas()
    raw_rows = len(flow_df)
    flow_df["slot"] = pd.to_datetime(flow_df["time_bin"], errors="coerce").dt.floor(cfg.TIME_INTERVAL)
    flow_df = flow_df.dropna(subset=["slot", "pulocationid", "dolocationid"])

    flow_slot = flow_df.groupby(["slot", "pulocationid", "dolocationid"], as_index=False)["flow_count"].sum()
    valid = (
        flow_slot["pulocationid"].isin(loc2ij)
        & flow_slot["dolocationid"].isin(loc2ij)
        & flow_slot["slot"].isin(time_map)
    )
    flow_valid = flow_slot[valid].copy()
    if flow_valid.empty:
        return raw_rows, 0

    pu_coords = flow_valid["pulocationid"].map(loc2ij)
    do_coords = flow_valid["dolocationid"].map(loc2ij)

    t_idx = flow_valid["slot"].map(time_map).to_numpy()
    i_idx = np.array([coord[0] for coord in pu_coords], dtype=np.int64)
    j_idx = np.array([coord[1] for coord in pu_coords], dtype=np.int64)
    k_idx = np.array([coord[0] for coord in do_coords], dtype=np.int64)
    l_idx = np.array([coord[1] for coord in do_coords], dtype=np.int64)
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


def load_normalization_max(volume, flow):
    eval_volume_max = float(volume.max())
    eval_flow_max = float(flow.max())

    if cfg.TRAIN_DATA_CONFIG_PATH.exists():
        with cfg.TRAIN_DATA_CONFIG_PATH.open("r", encoding="utf-8") as f:
            train_config = json.load(f)
        volume_max = float(train_config.get("raw_volume_train_max", eval_volume_max))
        flow_max = float(train_config.get("raw_flow_train_max", eval_flow_max))
        log(f"Using train scaler: volume={volume_max}, flow={flow_max}")
        return volume_max, flow_max, eval_volume_max, eval_flow_max

    log("Training data config not found, using eval raw max as fallback scaler")
    return eval_volume_max, eval_flow_max, eval_volume_max, eval_flow_max


def remove_obsolete_split_outputs():
    obsolete_names = (
        "volume_train.npz",
        "volume_test.npz",
        "flow_train.npz",
        "flow_test.npz",
        "context_train.npz",
        "context_test.npz",
        "data_test.json",
    )
    for name in obsolete_names:
        path = cfg.OUT_DIR / name
        if path.exists():
            path.unlink()
            log(f"Removed obsolete split output: {path}")


def save_eval_dataset(volume, flow, context, time_index, feature_names):
    time_str = time_index.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
    volume_max, flow_max, eval_volume_max, eval_flow_max = load_normalization_max(volume, flow)

    volume_norm = np.asarray(volume) / (volume_max + 1e-6)
    flow_norm = np.asarray(flow) / (flow_max + 1e-6)

    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    remove_obsolete_split_outputs()
    volume_path = cfg.OUT_DIR / "volume.npz"
    flow_path = cfg.OUT_DIR / "flow.npz"
    context_path = cfg.OUT_DIR / "context.npz"

    np.savez_compressed(volume_path, volume=volume_norm, time=time_str)
    log(f"Saved {volume_path}")
    np.savez_compressed(flow_path, flow=flow_norm, time=time_str)
    log(f"Saved {flow_path}")
    np.savez_compressed(context_path, context=context, time=time_str, feature_names=feature_names)
    log(f"Saved {context_path}")

    data_config = {
        "volume_test": str(volume_path),
        "flow_test": str(flow_path),
        "context_test": str(context_path),
        "volume_train_max": 1.0,
        "flow_train_max": 1.0,
        "raw_volume_train_max": volume_max,
        "raw_flow_train_max": flow_max,
        "eval_raw_volume_max": eval_volume_max,
        "eval_raw_flow_max": eval_flow_max,
        "timeslot_sec": cfg.TIMESLOT_SEC,
        "threshold": cfg.THRESHOLD,
        "context_dim": int(context.shape[1]),
        "context_feature_names": feature_names.tolist(),
        "start_time": time_str[0],
        "end_time": time_str[-1],
        "test_only_full": True,
    }
    with (cfg.OUT_DIR / "data.json").open("w", encoding="utf-8") as f:
        json.dump(data_config, f, indent=2)
    log(f"Saved {cfg.OUT_DIR / 'data.json'}")

    log(f"Volume eval: {volume_norm.shape}")
    log(f"Flow eval: {flow_norm.shape}")
    log(f"Context eval: {context.shape}")
    log("Time alignment check passed: volume_time == flow_time == context_time")


def main():
    log(f"Starting external eval dataset build for {cfg.START_TIME} -> {cfg.END_TIME}")
    log(f"Raw base dir: {cfg.RAW_BASE_DIR}")
    log(f"Output dir: {cfg.OUT_DIR}")
    log(f"Context: {cfg.CONTEXT_PATH}")
    validate_input_files()

    loc2ij = build_grid_mapping()
    log(f"Loaded {len(loc2ij)} location-grid mappings")

    time_index = build_time_index()
    time_map = {t: i for i, t in enumerate(time_index)}
    context, feature_names = load_context_for_times(time_index)

    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    volume_temp_path = cfg.OUT_DIR / "eval_volume_temp.dat"
    flow_temp_path = cfg.OUT_DIR / "eval_flow_temp.dat"
    volume = np.memmap(
        volume_temp_path,
        dtype=np.float32,
        mode="w+",
        shape=(len(time_index), cfg.GRID_H, cfg.GRID_W, 2),
    )
    flow = np.memmap(
        flow_temp_path,
        dtype=np.float32,
        mode="w+",
        shape=(2, len(time_index), cfg.GRID_H, cfg.GRID_W, cfg.GRID_H, cfg.GRID_W),
    )

    for month in cfg.MONTHS:
        volume_path, flow_path = month_paths(month)
        log(f"Processing {cfg.YEAR}-{month:02d}")
        log(f"  volume path: {volume_path}")
        vol_raw, vol_valid = process_volume_month(volume_path, volume, time_map, loc2ij)
        log(f"  volume rows raw/valid: {vol_raw}/{vol_valid}")
        log(f"  flow path: {flow_path}")
        flow_raw, flow_valid = process_flow_month(flow_path, flow, time_map, loc2ij)
        log(f"  flow rows raw/valid: {flow_raw}/{flow_valid}")

    log(f"Nonzero volume: {int(np.count_nonzero(volume))}")
    log(f"Nonzero flow: {int(np.count_nonzero(flow))}")
    save_eval_dataset(volume, flow, context, time_index, feature_names)

    del volume, flow
    for temp_path in (volume_temp_path, flow_temp_path):
        try:
            temp_path.unlink()
            log(f"Deleted temp file: {temp_path}")
        except FileNotFoundError:
            pass

    log("Done")


if __name__ == "__main__":
    main()
