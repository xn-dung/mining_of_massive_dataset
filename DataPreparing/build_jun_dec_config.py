from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]

# Edit these paths when moving between local and Colab.
RAW_BASE_DIR = WORKSPACE_ROOT / "Data_V_F" / "Raw_Test"
LOOKUP_PATH = WORKSPACE_ROOT / "Data_V_F" / "taxi_zone_lookup_grid.csv"
CONTEXT_PATH = WORKSPACE_ROOT / "MockData" / "context_2025.npz"
OUT_DIR = WORKSPACE_ROOT / "Data_V_F" / "Prepare_Test_2025"

# Dataset settings.
DATASET = "yellow"
YEAR = 2025
MONTHS = list(range(1, 4))

GRID_H = 10
GRID_W = 20
TIME_INTERVAL = "30min"
TIMESLOT_SEC = 1800

START_TIME = f"{YEAR}-01-01 00:00:00"
END_TIME = f"{YEAR}-03-31 23:30:00"

TRAIN_SPLIT = 0.8
TEST_ONLY_FULL = False
THRESHOLD = 0
