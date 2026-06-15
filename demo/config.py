from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = PROJECT_ROOT / "demo"

DATA_DIR = DEMO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INITIAL_HISTORY_DIR = RAW_DIR / "initial_history"
CRAWLED_DIR = DATA_DIR / "crawled"
PROCESSED_DIR = DATA_DIR / "processed" / "stdn"
INITIAL_HISTORY_PROCESSED_DIR = PROCESSED_DIR / "initial_history"
INCOMING_ACTUAL_DIR = DATA_DIR / "incoming" / "actual"

ARTIFACT_DIR = DEMO_ROOT / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
PREDICTION_DIR = ARTIFACT_DIR / "predictions"
METRIC_DIR = ARTIFACT_DIR / "metrics"
MLFLOW_DIR = ARTIFACT_DIR / "mlruns"
SQLITE_PATH = ARTIFACT_DIR / "mlops.sqlite"
ACTIVE_MODEL_FILE = MODEL_DIR / "active_model.txt"

TRAINING_DIR = PROJECT_ROOT / "Training_Eval_Phase"
STDN_DIR = PROJECT_ROOT / "STDN"
CRAWL_STDN_DIR = DEMO_ROOT / "CrawlSTDN"
CRAWL_STDN_CONFIG_PATH = CRAWL_STDN_DIR / "configSTDN.yaml"
INITIAL_RAW_SOURCE_DIR = PROJECT_ROOT / "Data_V_F" / "Raw"
INITIAL_CONTEXT_SOURCE_PATH = PROJECT_ROOT / "Data_V_F" / "prepare_6-12_2024" / "context_test.npz"
TRAIN_DATA_CONFIG_PATH = PROJECT_ROOT / "Data_V_F" / "prepare_6-12_2024" / "data.json"
INITIAL_HISTORY_MONTH = "12"
INITIAL_HISTORY_PERIOD = "2024-12"
GRID_LOOKUP_SOURCE_PATH = PROJECT_ROOT / "Data_V_F" / "taxi_zone_lookup_grid.csv"
ZONE_LOOKUP_SOURCE_PATH = PROJECT_ROOT / "Data_V_F" / "taxi_zone_lookup.csv"
GRID_LOOKUP_PATH = RAW_DIR / "taxi_zone_lookup_grid.csv"
ZONE_LOOKUP_PATH = RAW_DIR / "taxi_zone_lookup.csv"
LOOKUP_PATH = GRID_LOOKUP_PATH

INITIAL_MODEL_VERSION = "stdn_v1"
INITIAL_MODEL_PATH = MODEL_DIR / f"{INITIAL_MODEL_VERSION}.pth"

# Metrics in this demo are reported as percentages. 25.0 means 25% WMAPE.
INITIAL_LAST_UPDATE_DATE = "2024-12-31"
WMAPE_THRESHOLD = 25.0
PREDICTION_REFRESH_DAYS = 30
EVALUATION_INTERVAL_DAYS = 90
EVALUATION_WINDOW_DAYS = 7
DEMO_TRAIN_EPOCHS = 1

TIME_INTERVAL = "30min"
TIMESLOT_SEC = 1800
GRID_H = 10
GRID_W = 20
OUTPUT_SHAPE = 2


def ensure_demo_dirs():
    for path in (
        RAW_DIR,
        INITIAL_HISTORY_DIR,
        CRAWLED_DIR,
        PROCESSED_DIR,
        INCOMING_ACTUAL_DIR,
        MODEL_DIR,
        PREDICTION_DIR,
        METRIC_DIR,
        MLFLOW_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
