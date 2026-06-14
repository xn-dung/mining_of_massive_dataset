from pathlib import Path


# Colab path. If this path is not available, scripts fall back to the local repo.
COLAB_PROJECT_ROOT = Path("/content/drive/MyDrive/Mining_ofMasssive")
LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = COLAB_PROJECT_ROOT if COLAB_PROJECT_ROOT.exists() else LOCAL_PROJECT_ROOT

STDN_DIR = PROJECT_ROOT / "STDN"
TRAINING_DIR = PROJECT_ROOT / "Training_Eval_Phase"
DATA_ROOT = PROJECT_ROOT / "Data_V_F" / "prepare_6-12_2024"
EVAL_DATA_ROOT = PROJECT_ROOT / "Data_V_F" / "Prepare_Test_2025"

SOURCE_DATA_CONFIG = DATA_ROOT / "data.json"
RUNTIME_DATA_CONFIG = TRAINING_DIR / "runtime_data.json"
MODEL_SAVE_PATH = TRAINING_DIR / "stdn_context_jun_dec_2024.pth"
TRAIN_TEST_PREDICTIONS_PATH = TRAINING_DIR / "stdn_context_train_test_predictions.npz"
TRAIN_TEST_METRICS_PATH = TRAINING_DIR / "stdn_context_train_test_metrics.json"
EVAL_PREDICTIONS_PATH = TRAINING_DIR / "stdn_context_eval_predictions.npz"
EVAL_METRICS_PATH = TRAINING_DIR / "stdn_context_eval_metrics.json"


# Dataset / model window settings.
ATT_LSTM_NUM = 3
LONG_TERM_LSTM_SEQ_LEN = 3
SHORT_TERM_LSTM_SEQ_LEN = 7
HIST_FEATURE_DAYNUM = 7
NBHD_SIZE = 2
CNN_NBHD_SIZE = 3


# Training settings.
BATCH_SIZE = 64
MAX_EPOCHS = 2
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
NUM_WORKERS = 0
PRINT_EVERY = 25
PIN_MEMORY = True


# Model settings.
CNN_FLAT_SIZE = 128
LSTM_OUT_SIZE = 128
DROPOUT_RATE = 0.5
OUTPUT_SHAPE = 2


# Evaluation settings.
MAPE_EPS = 1e-6
FILTERED_MAPE_THRESHOLD = 10.0
