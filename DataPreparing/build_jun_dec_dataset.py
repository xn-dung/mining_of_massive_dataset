import numpy as np

import build_jun_dec_config as cfg
from build_jun_dec_preprocess import (
    build_time_index,
    load_context_for_times,
    load_location_mapping,
    log,
    month_paths,
    process_flow_month,
    process_volume_month,
    save_split,
    validate_input_files,
)


def main():
    log(f"Starting STDN dataset build for {cfg.START_TIME} -> {cfg.END_TIME}")
    log(f"Raw base dir: {cfg.RAW_BASE_DIR}")
    log(f"Output dir: {cfg.OUT_DIR}")
    log(f"Context: {cfg.CONTEXT_PATH}")
    log(f"Lookup: {cfg.LOOKUP_PATH}")

    validate_input_files()

    loc2ij = load_location_mapping()
    time_index = build_time_index()
    time_map = {t: i for i, t in enumerate(time_index)}
    context, feature_names = load_context_for_times(time_index)

    log(f"Time slots: {len(time_index)} ({time_index[0]} -> {time_index[-1]})")

    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    volume_path = cfg.OUT_DIR / "volume_temp.dat"
    flow_path = cfg.OUT_DIR / "flow_temp.dat"

    volume = np.memmap(
        volume_path,
        dtype=np.float32,
        mode="w+",
        shape=(len(time_index), cfg.GRID_H, cfg.GRID_W, 2),
    )
    flow = np.memmap(
        flow_path,
        dtype=np.float32,
        mode="w+",
        shape=(2, len(time_index), cfg.GRID_H, cfg.GRID_W, cfg.GRID_H, cfg.GRID_W),
    )

    for month in cfg.MONTHS:
        vol_path, flow_path_month = month_paths(month)
        log(f"Processing {cfg.YEAR}-{month:02d}")
        log(f"  volume path: {vol_path}")
        vol_raw, vol_valid = process_volume_month(vol_path, volume, time_map, loc2ij)
        log(f"  volume rows raw/valid: {vol_raw}/{vol_valid}")
        log(f"  flow path: {flow_path_month}")
        flow_raw, flow_valid = process_flow_month(flow_path_month, flow, time_map, loc2ij)
        log(f"  flow rows raw/valid: {flow_raw}/{flow_valid}")

    log(f"Nonzero volume: {int(np.count_nonzero(volume))}")
    log(f"Nonzero flow: {int(np.count_nonzero(flow))}")

    save_split(volume, flow, context, time_index, feature_names)

    del volume, flow
    for tmp_path in (volume_path, flow_path):
        try:
            tmp_path.unlink()
            log(f"Deleted temp file: {tmp_path}")
        except FileNotFoundError:
            pass

    log("Done")


if __name__ == "__main__":
    main()
