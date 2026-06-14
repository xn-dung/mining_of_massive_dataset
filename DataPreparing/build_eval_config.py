from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]

# Raw eval data. This dataset is for external evaluation only, so it is not split.
RAW_BASE_DIR = WORKSPACE_ROOT / "Data_V_F" / "Raw_Test"
LOOKUP_PATH = WORKSPACE_ROOT / "Data_V_F" / "taxi_zone_lookup_grid.csv"
CONTEXT_PATH = WORKSPACE_ROOT / "MockData" / "context_2025.npz"
OUT_DIR = WORKSPACE_ROOT / "Data_V_F" / "Prepare_Test_2025"

# Use the training dataset scaler to normalize external eval data consistently
# with the checkpoint that was trained on Jun-Dec 2024.
TRAIN_DATA_CONFIG_PATH = WORKSPACE_ROOT / "Data_V_F" / "prepare_6-12_2024" / "data.json"

DATASET = "yellow"
YEAR = 2025
MONTHS = list(range(1, 4))

GRID_H = 10
GRID_W = 20
TIME_INTERVAL = "30min"
TIMESLOT_SEC = 1800

START_TIME = f"{YEAR}-01-01 00:00:00"
END_TIME = f"{YEAR}-03-31 23:30:00"

THRESHOLD = 0
