from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..utils.file_utils import read_table


def _find_time_column(df):
    for col in ("time", "time_bin", "datetime", "timestamp", "date"):
        if col in df.columns:
            return col
    return None


def _as_paths(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [Path(path) for path in value if path and Path(path).exists()]
    path = Path(value)
    return [path] if path.exists() else []


def build_context_features(weather_path=None, holiday_path=None, time_index=None, output_path=None):
    time_index = pd.DatetimeIndex(time_index)
    features = pd.DataFrame(index=time_index)

    weather_frames = []
    for path in _as_paths(weather_path):
        weather = read_table(path)
        time_col = _find_time_column(weather)
        if time_col:
            weather[time_col] = pd.to_datetime(weather[time_col], errors="coerce").dt.floor(config.TIME_INTERVAL)
            weather_frames.append(weather.dropna(subset=[time_col]).set_index(time_col))

    if weather_frames:
        weather = pd.concat(weather_frames).sort_index()
        weather = weather[~weather.index.duplicated(keep="last")]
        numeric_weather = weather.select_dtypes(include="number")
        if not numeric_weather.empty:
            features = features.join(numeric_weather.groupby(level=0).mean(), how="left")

    holiday_dates = set()
    for path in _as_paths(holiday_path):
        holiday = read_table(path)
        date_col = _find_time_column(holiday)
        if date_col:
            holiday[date_col] = pd.to_datetime(holiday[date_col], errors="coerce").dt.date
            holiday_dates.update(holiday.dropna(subset=[date_col])[date_col])

    if holiday_dates:
        features["is_holiday"] = [1.0 if ts.date() in holiday_dates else 0.0 for ts in time_index]

    if features.empty:
        features["bias"] = 0.0

    features = features.reindex(time_index).ffill().bfill().fillna(0.0)
    feature_names = np.asarray(features.columns.astype(str), dtype=str)
    context = features.to_numpy(dtype=np.float32)
    output_path = Path(output_path or config.PROCESSED_DIR / "context.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        context=context,
        time=time_index.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
        feature_names=feature_names,
    )
    return output_path
