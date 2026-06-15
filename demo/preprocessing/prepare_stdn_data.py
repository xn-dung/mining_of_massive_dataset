import gc
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import config
from ..preprocessing.build_context_features import build_context_features
from ..utils.file_utils import ensure_dir, read_json


ARRAY_KEYS = ("volume_train", "volume_test", "flow_train", "flow_test", "context_train", "context_test")


def _log(message):
    print(f"[demo preprocess] {message}", flush=True)


def _chunk_slices(length, chunk_size=100_000):
    for start in range(0, length, chunk_size):
        yield slice(start, min(start + chunk_size, length))


def _copy_existing_stdn_dataset(source_dir, output_dir):
    source_dir = Path(source_dir)
    output_dir = ensure_dir(output_dir)
    for path in source_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".json", ".npz", ".npy"}:
            shutil.copy2(path, output_dir / path.name)

    data_config = read_json(output_dir / "data.json")
    if data_config:
        for key in ARRAY_KEYS:
            if key in data_config:
                data_config[key] = str(output_dir / Path(data_config[key]).name)
        with (output_dir / "data.json").open("w", encoding="utf-8") as f:
            json.dump(data_config, f, indent=2)
    return output_dir


def _find_file(input_dir, keywords, suffixes):
    matches = _find_files(input_dir, keywords, suffixes)
    return matches[0] if matches else None


def _find_files(input_dir, keywords, suffixes):
    input_dir = Path(input_dir)
    matches = []
    for path in input_dir.iterdir():
        name = path.name.lower()
        if any(keyword in name for keyword in keywords) and path.suffix.lower() in suffixes:
            matches.append(path)
    return sorted(matches)


def _build_grid_mapping():
    grid = pd.read_csv(config.LOOKUP_PATH)
    grid["Grid_X"] = grid["Grid_X"].astype(int)
    grid["Grid_Y"] = grid["Grid_Y"].astype(int)
    return dict(zip(grid["LocationID"], zip(grid["Grid_X"], grid["Grid_Y"])))


def _build_time_index(volume_paths, start_time=None, end_time=None):
    if start_time is not None and end_time is not None:
        return pd.date_range(pd.Timestamp(start_time), pd.Timestamp(end_time), freq=config.TIME_INTERVAL)

    all_slots = []
    for volume_path in volume_paths:
        volume_df = pd.read_csv(volume_path, usecols=["time_bin"])
        slots = pd.to_datetime(volume_df["time_bin"], errors="coerce").dropna().dt.floor(config.TIME_INTERVAL)
        if start_time is not None:
            slots = slots[slots >= pd.Timestamp(start_time)]
        if end_time is not None:
            slots = slots[slots <= pd.Timestamp(end_time)]
        all_slots.append(slots)
    slots = pd.concat(all_slots, ignore_index=True) if all_slots else pd.Series(dtype="datetime64[ns]")
    if slots.empty:
        raise ValueError("No valid time_bin values found in STDN volume files")
    return pd.date_range(slots.min(), slots.max(), freq=config.TIME_INTERVAL)


def _process_volume(volume_path, volume, time_map, loc2ij):
    _log(f"Loading volume: {volume_path}")
    vol_df = pd.read_csv(volume_path)
    vol_df["slot"] = pd.to_datetime(vol_df["time_bin"], errors="coerce").dt.floor(config.TIME_INTERVAL)
    location_col = "locationid" if "locationid" in vol_df.columns else "LocationID"
    vol_df["LocationID"] = pd.to_numeric(vol_df[location_col], errors="coerce")
    vol_df = vol_df.dropna(subset=["slot", "LocationID"])
    vol_df["LocationID"] = vol_df["LocationID"].astype(int)

    grouped = vol_df.groupby(["slot", "LocationID"], as_index=False)[["start_volume", "end_volume"]].sum()
    valid = grouped["LocationID"].isin(loc2ij) & grouped["slot"].isin(time_map)
    grouped = grouped[valid]
    if grouped.empty:
        return

    coords = grouped["LocationID"].map(loc2ij)
    t_idx = grouped["slot"].map(time_map).to_numpy()
    x_idx = np.array([coord[0] for coord in coords], dtype=np.int64)
    y_idx = np.array([coord[1] for coord in coords], dtype=np.int64)
    starts = grouped["start_volume"].to_numpy(dtype=np.float32)
    ends = grouped["end_volume"].to_numpy(dtype=np.float32)

    for slc in tqdm(list(_chunk_slices(len(grouped))), desc="Build volume grid", dynamic_ncols=True):
        np.add.at(volume[..., 0], (t_idx[slc], x_idx[slc], y_idx[slc]), starts[slc])
        np.add.at(volume[..., 1], (t_idx[slc], x_idx[slc], y_idx[slc]), ends[slc])
    _log(f"Volume rows mapped: {len(grouped)}")


def _process_flow(flow_path, flow, time_map, loc2ij):
    import pyarrow.parquet as pq

    _log(f"Loading flow: {flow_path}")
    pf = pq.ParquetFile(flow_path)
    flow_df = pf.read(columns=["time_bin", "pulocationid", "dolocationid", "flow_count"]).to_pandas()
    flow_df["slot"] = pd.to_datetime(flow_df["time_bin"], errors="coerce").dt.floor(config.TIME_INTERVAL)
    flow_df = flow_df.dropna(subset=["slot", "pulocationid", "dolocationid"])

    grouped = flow_df.groupby(["slot", "pulocationid", "dolocationid"], as_index=False)["flow_count"].sum()
    valid = (
        grouped["pulocationid"].isin(loc2ij)
        & grouped["dolocationid"].isin(loc2ij)
        & grouped["slot"].isin(time_map)
    )
    grouped = grouped[valid]
    if grouped.empty:
        return

    pu_coords = grouped["pulocationid"].map(loc2ij)
    do_coords = grouped["dolocationid"].map(loc2ij)
    t_idx = grouped["slot"].map(time_map).to_numpy()
    x_idx = np.array([coord[0] for coord in pu_coords], dtype=np.int64)
    y_idx = np.array([coord[1] for coord in pu_coords], dtype=np.int64)
    k_idx = np.array([coord[0] for coord in do_coords], dtype=np.int64)
    l_idx = np.array([coord[1] for coord in do_coords], dtype=np.int64)
    counts = grouped["flow_count"].to_numpy(dtype=np.float32)

    for slc in tqdm(list(_chunk_slices(len(grouped))), desc="Build flow grid", dynamic_ncols=True):
        np.add.at(flow[0], (t_idx[slc], x_idx[slc], y_idx[slc], k_idx[slc], l_idx[slc]), counts[slc])

        prev_mask = t_idx[slc] > 0
        if np.any(prev_mask):
            np.add.at(
                flow[1],
                (
                    t_idx[slc][prev_mask] - 1,
                    x_idx[slc][prev_mask],
                    y_idx[slc][prev_mask],
                    k_idx[slc][prev_mask],
                    l_idx[slc][prev_mask],
                ),
                counts[slc][prev_mask],
            )
    _log(f"Flow rows mapped: {len(grouped)}")


def _normalization_max(volume, flow):
    eval_volume_max = float(np.max(volume) or 1.0)
    eval_flow_max = float(np.max(flow) or 1.0)

    if config.TRAIN_DATA_CONFIG_PATH.exists():
        train_config = read_json(config.TRAIN_DATA_CONFIG_PATH, default={}) or {}
        volume_max = float(train_config.get("raw_volume_train_max", eval_volume_max))
        flow_max = float(train_config.get("raw_flow_train_max", eval_flow_max))
        return volume_max, flow_max, eval_volume_max, eval_flow_max

    return eval_volume_max, eval_flow_max, eval_volume_max, eval_flow_max


def _close_memmap(array):
    try:
        array.flush()
    except AttributeError:
        return
    mmap_obj = getattr(array, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()


def _load_aligned_context(context_path, time_index):
    context_paths = [Path(path) for path in context_path] if isinstance(context_path, (list, tuple)) else [Path(context_path)]
    frames = []
    feature_names = None

    for path in context_paths:
        context_npz = np.load(path, allow_pickle=True)
        context = context_npz["context"].astype(np.float32)
        names = context_npz["feature_names"].astype(str)
        if feature_names is None:
            feature_names = names
        elif list(feature_names) != list(names):
            raise ValueError(f"Context feature names do not match across STDN context files: {path}")

        if "time" not in context_npz.files:
            if len(context_paths) > 1:
                raise ValueError(f"Context file has no time array and cannot be merged with others: {path}")
            if len(context) != len(time_index):
                raise ValueError(
                    f"Context length {len(context)} does not match time index length {len(time_index)} "
                    f"and no time array is available: {path}"
                )
            return context, feature_names
        context_time = pd.to_datetime(context_npz["time"])
        frames.append(pd.DataFrame(context, index=context_time, columns=names))

    context_df = pd.concat(frames).sort_index()
    context_df = context_df[~context_df.index.duplicated(keep="last")]
    aligned = context_df.reindex(time_index)
    missing = aligned[aligned.isna().any(axis=1)].index
    if len(missing) > 0:
        raise ValueError(
            "Context is missing expected time bins. First missing bins: "
            + ", ".join(str(t) for t in missing[:10])
        )
    return aligned.to_numpy(dtype=np.float32), feature_names


def _write_stdn_dataset(output_dir, volume, flow, context_path, time_index, split_train=False):
    output_dir = ensure_dir(output_dir)
    _log(f"Aligning context: {context_path}")
    context, feature_names = _load_aligned_context(context_path, time_index)
    time_str = time_index.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
    volume_max, flow_max, eval_volume_max, eval_flow_max = _normalization_max(volume, flow)
    volume_norm = np.asarray(volume) / (volume_max + 1e-6)
    flow_norm = np.asarray(flow) / (flow_max + 1e-6)

    if split_train:
        split = int(len(time_index) * 0.8)
        np.savez_compressed(output_dir / "volume_train.npz", volume=volume_norm[:split], time=time_str[:split])
        np.savez_compressed(output_dir / "volume_test.npz", volume=volume_norm[split:], time=time_str[split:])
        np.savez_compressed(output_dir / "flow_train.npz", flow=flow_norm[:, :split], time=time_str[:split])
        np.savez_compressed(output_dir / "flow_test.npz", flow=flow_norm[:, split:], time=time_str[split:])
        np.savez_compressed(output_dir / "context_train.npz", context=context[:split], time=time_str[:split], feature_names=feature_names)
        np.savez_compressed(output_dir / "context_test.npz", context=context[split:], time=time_str[split:], feature_names=feature_names)
    else:
        _log("Saving processed STDN test arrays")
        np.savez_compressed(output_dir / "volume_test.npz", volume=volume_norm, time=time_str)
        np.savez_compressed(output_dir / "flow_test.npz", flow=flow_norm, time=time_str)
        np.savez_compressed(output_dir / "context_test.npz", context=context, time=time_str, feature_names=feature_names)

    data_config = {
        "volume_train": str(output_dir / ("volume_train.npz" if split_train else "volume_test.npz")),
        "volume_test": str(output_dir / "volume_test.npz"),
        "flow_train": str(output_dir / ("flow_train.npz" if split_train else "flow_test.npz")),
        "flow_test": str(output_dir / "flow_test.npz"),
        "context_train": str(output_dir / ("context_train.npz" if split_train else "context_test.npz")),
        "context_test": str(output_dir / "context_test.npz"),
        "volume_train_max": 1.0,
        "flow_train_max": 1.0,
        "raw_volume_train_max": volume_max,
        "raw_flow_train_max": flow_max,
        "eval_raw_volume_max": eval_volume_max,
        "eval_raw_flow_max": eval_flow_max,
        "timeslot_sec": config.TIMESLOT_SEC,
        "threshold": 0,
        "context_dim": int(context.shape[1]),
        "context_feature_names": feature_names.tolist(),
        "start_time": str(time_str[0]),
        "end_time": str(time_str[-1]),
        "test_only_full": not split_train,
    }
    with (output_dir / "data.json").open("w", encoding="utf-8") as f:
        json.dump(data_config, f, indent=2)
    _log(f"Saved processed dataset: {output_dir}")
    return output_dir


def prepare_stdn_data(input_dir, output_dir, split_train=False, start_time=None, end_time=None):
    input_dir = Path(input_dir)
    output_dir = ensure_dir(output_dir)

    if (input_dir / "data.json").exists():
        return _copy_existing_stdn_dataset(input_dir, output_dir)

    volume_npz = _find_file(input_dir, ("volume_test", "volume"), {".npz"})
    flow_npz = _find_file(input_dir, ("flow_test", "flow"), {".npz"})
    context_npz = _find_file(input_dir, ("context_test", "context"), {".npz"})
    if volume_npz and flow_npz and context_npz:
        shutil.copy2(volume_npz, output_dir / "volume_test.npz")
        shutil.copy2(flow_npz, output_dir / "flow_test.npz")
        shutil.copy2(context_npz, output_dir / "context_test.npz")
        volume = np.load(output_dir / "volume_test.npz", allow_pickle=True)["volume"]
        flow = np.load(output_dir / "flow_test.npz", allow_pickle=True)["flow"]
        time_index = pd.to_datetime(np.load(output_dir / "volume_test.npz", allow_pickle=True)["time"])
        mask = np.ones(len(time_index), dtype=bool)
        if start_time is not None:
            mask &= time_index >= pd.Timestamp(start_time)
        if end_time is not None:
            mask &= time_index <= pd.Timestamp(end_time)
        volume = volume[mask]
        flow = flow[:, mask]
        time_index = time_index[mask]
        return _write_stdn_dataset(output_dir, volume, flow, output_dir / "context_test.npz", time_index, split_train)

    volume_paths = _find_files(input_dir, ("volume",), {".csv"})
    flow_paths = _find_files(input_dir, ("flow",), {".parquet", ".pq"})
    if not volume_paths or not flow_paths:
        raise FileNotFoundError("Expected either STDN data.json/npz files or raw volume.csv + flow.parquet")

    time_index = _build_time_index(volume_paths, start_time=start_time, end_time=end_time)
    _log(f"Preparing STDN data for {time_index[0]} -> {time_index[-1]} ({len(time_index)} slots)")
    time_map = {time: idx for idx, time in enumerate(time_index)}
    loc2ij = _build_grid_mapping()
    volume_temp_path = output_dir / "volume_temp.dat"
    flow_temp_path = output_dir / "flow_temp.dat"
    volume = np.memmap(
        volume_temp_path,
        dtype=np.float32,
        mode="w+",
        shape=(len(time_index), config.GRID_H, config.GRID_W, config.OUTPUT_SHAPE),
    )
    flow = np.memmap(
        flow_temp_path,
        dtype=np.float32,
        mode="w+",
        shape=(2, len(time_index), config.GRID_H, config.GRID_W, config.GRID_H, config.GRID_W),
    )

    try:
        for volume_path in volume_paths:
            _process_volume(volume_path, volume, time_map, loc2ij)
        for flow_path in flow_paths:
            _process_flow(flow_path, flow, time_map, loc2ij)

        context_paths = _find_files(input_dir, ("context",), {".npz"})
        if context_paths:
            context_path = context_paths
        else:
            context_path = build_context_features(
                weather_path=_find_files(input_dir, ("weather",), {".csv", ".parquet", ".pq"}),
                holiday_path=_find_files(input_dir, ("holiday",), {".csv", ".parquet", ".pq"}),
                time_index=time_index,
                output_path=output_dir / "context_test.npz",
            )
        return _write_stdn_dataset(output_dir, volume, flow, context_path, time_index, split_train)
    finally:
        _close_memmap(volume)
        _close_memmap(flow)
        del volume, flow
        gc.collect()
        for temp_path in (volume_temp_path, flow_temp_path):
            try:
                temp_path.unlink()
            except (FileNotFoundError, PermissionError):
                pass
