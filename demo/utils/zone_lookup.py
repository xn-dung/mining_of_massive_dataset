from pathlib import Path

import pandas as pd

from .. import config


def lookup_exists():
    return Path(config.GRID_LOOKUP_PATH).exists()


def load_grid_lookup():
    path = Path(config.GRID_LOOKUP_PATH)
    if not path.exists():
        return pd.DataFrame()
    grid = pd.read_csv(path)
    rename = {
        "Grid_X": "grid_x",
        "Grid_Y": "grid_y",
        "LocationID": "location_id",
    }
    grid = grid.rename(columns={key: value for key, value in rename.items() if key in grid.columns})
    required = {"grid_x", "grid_y", "location_id"}
    if not required.issubset(grid.columns):
        return pd.DataFrame()
    return grid


def load_zone_lookup():
    path = Path(config.ZONE_LOOKUP_PATH)
    if not path.exists():
        return pd.DataFrame()
    zones = pd.read_csv(path)
    rename = {
        "LocationID": "location_id",
        "Zone": "zone_name",
        "Borough": "borough",
    }
    zones = zones.rename(columns={key: value for key, value in rename.items() if key in zones.columns})
    required = {"location_id", "zone_name", "borough"}
    if not required.issubset(zones.columns):
        return pd.DataFrame()
    return zones[["location_id", "borough", "zone_name"]]


def grid_zone_summary():
    grid = load_grid_lookup()
    if grid.empty:
        return grid

    zones = load_zone_lookup()
    if not zones.empty:
        lookup = grid.merge(zones, on="location_id", how="left")
    else:
        lookup = grid.copy()
        lookup["borough"] = ""
        lookup["zone_name"] = ""

    grouped = lookup.groupby(["grid_x", "grid_y"], as_index=False).agg(
        zone_name=("zone_name", lambda values: " / ".join(sorted(set(str(v) for v in values if pd.notna(v)))))
        if "zone_name" in lookup.columns
        else ("grid_x", lambda values: ""),
        borough=("borough", lambda values: " / ".join(sorted(set(str(v) for v in values if pd.notna(v)))))
        if "borough" in lookup.columns
        else ("grid_x", lambda values: ""),
    )
    return grouped


def enrich_prediction_frame(prediction_df):
    zones = grid_zone_summary()
    if zones.empty:
        return prediction_df
    base = prediction_df.drop(columns=[col for col in ("borough", "zone_name") if col in prediction_df.columns])
    if not {"grid_x", "grid_y"}.issubset(base.columns):
        return prediction_df
    return base.merge(zones, on=["grid_x", "grid_y"], how="left")
