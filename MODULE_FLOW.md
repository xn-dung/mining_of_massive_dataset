# STDN Context Pipeline Guide

Tai lieu nay mo ta luong chay cac module cho STDN co context.

## 1. Tao `context.npz`

File dung:

```text
MockData/build_context_features.ipynb
```

Input can co:

```text
MockData/holidays_2024.csv
MockData/weather_2024.csv
```

hoac:

```text
MockData/holidays_2025.csv
MockData/weather_2025.csv
```

Trong notebook, sua:

```python
YEAR = 2025
HOLIDAY_PATH = BASE_DIR / "holidays_2025.csv"
WEATHER_PATH = BASE_DIR / "weather_2025.csv"
```

Output:

```text
MockData/context_2025.csv
MockData/context_2025.npz
MockData/context_2025_meta.json
```

Context dang la additive context, neutral slot gan `0.0`, dung de cong vao feature sau `volume * sigmoid(flow)`.

## 2. Tao volume/flow/context co chia train/test

Module dung cho training dataset.

Files:

```text
DataPreparing/build_jun_dec_config.py
DataPreparing/build_jun_dec_preprocess.py
DataPreparing/build_jun_dec_dataset.py
```

Sua config:

```python
RAW_BASE_DIR = WORKSPACE_ROOT / "Data_V_F" / "Raw"
CONTEXT_PATH = WORKSPACE_ROOT / "MockData" / "context_2024.npz"
OUT_DIR = WORKSPACE_ROOT / "Data_V_F" / "prepare_6-12_2024"

YEAR = 2024
MONTHS = list(range(6, 13))
START_TIME = f"{YEAR}-06-01 00:00:00"
END_TIME = f"{YEAR}-12-31 23:30:00"

TRAIN_SPLIT = 0.8
TEST_ONLY_FULL = False
```

Chay:

```bash
python DataPreparing/build_jun_dec_dataset.py
```

Output:

```text
volume_train.npz
volume_test.npz
flow_train.npz
flow_test.npz
context_train.npz
context_test.npz
data.json
```

Luon check trong log:

```text
Time alignment check passed inside each split: volume_time == flow_time == context_time
```

## 3. Tao volume/flow/context external eval khong chia train/test

Module dung cho data moi hoan toan, vi du 2025 Jan-Mar.

Files:

```text
DataPreparing/build_eval_config.py
DataPreparing/build_eval_dataset.py
```

Sua config:

```python
RAW_BASE_DIR = WORKSPACE_ROOT / "Data_V_F" / "Raw_Test"
CONTEXT_PATH = WORKSPACE_ROOT / "MockData" / "context_2025.npz"
OUT_DIR = WORKSPACE_ROOT / "Data_V_F" / "Prepare_Test_2025"

TRAIN_DATA_CONFIG_PATH = WORKSPACE_ROOT / "Data_V_F" / "prepare_6-12_2024" / "data.json"

YEAR = 2025
MONTHS = list(range(1, 4))
START_TIME = f"{YEAR}-01-01 00:00:00"
END_TIME = f"{YEAR}-03-31 23:30:00"
```

Chay:

```bash
python DataPreparing/build_eval_dataset.py
```

Output khong chia split:

```text
volume.npz
flow.npz
context.npz
data.json
```

Trong `data.json`, cac key `volume_test`, `flow_test`, `context_test` se tro vao 3 file tren.

Luu y quan trong:

```text
build_eval_dataset.py uu tien dung raw_volume_train_max/raw_flow_train_max tu training data.json.
```

Dieu nay giup external eval data duoc normalize cung scale voi checkpoint da train.

## 4. Training + validation + held-out test

Files:

```text
Training_Eval_Phase/train_config.py
Training_Eval_Phase/train_utils.py
Training_Eval_Phase/train_context_stdn.py
```

Sua config training:

```python
DATA_ROOT = PROJECT_ROOT / "Data_V_F" / "prepare_6-12_2024"
EVAL_DATA_ROOT = PROJECT_ROOT / "Data_V_F" / "Prepare_Test_2025"
BATCH_SIZE = 64
MAX_EPOCHS = 2
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
```

Chay:

```bash
python Training_Eval_Phase/train_context_stdn.py
```

Flow trong file train:

```text
1. Doc volume_train/flow_train/context_train tu data.json
2. Chia train/valid theo thoi gian, khong random theo time slot
3. Train tren train subset
4. Validate sau moi epoch voi RMSE, WMAPE, Filtered MAPE, R2
5. Save best checkpoint theo val_loss
6. Load best checkpoint
7. Test tren held-out volume_test/flow_test/context_test cua training dataset
8. Save prediction + metric
```

Output:

```text
Training_Eval_Phase/stdn_context_jun_dec_2024.pth
Training_Eval_Phase/stdn_context_train_test_predictions.npz
Training_Eval_Phase/stdn_context_train_test_metrics.json
```

## 5. External eval tren data moi

Files:

```text
Training_Eval_Phase/eval_context_stdn.py
Training_Eval_Phase/train_utils.py
```

Chay bang Python:

```bash
python Training_Eval_Phase/eval_context_stdn.py
```

Mac dinh no dung:

```python
cfg.EVAL_DATA_ROOT
```

Neu muon override data root khac, dung trong notebook hoac script:

```python
import eval_context_stdn
import train_config as cfg

eval_data_root = cfg.PROJECT_ROOT / "Data_V_F" / "Prepare_Test_2025"
eval_results = eval_context_stdn.run_eval(cfg, data_root=eval_data_root)
```

Flow trong file eval:

```text
1. Doc external data root rieng
2. Doc data.json cua external eval dataset
3. Tao test loader tu volume_test/flow_test/context_test
4. Load checkpoint tu cfg.MODEL_SAVE_PATH
5. Eval RMSE, WMAPE, Filtered MAPE, R2
6. Save prediction + metric
```

Output:

```text
Training_Eval_Phase/stdn_context_eval_predictions.npz
Training_Eval_Phase/stdn_context_eval_metrics.json
```

## 6. Notebook Colab

Notebook:

```text
Training_Eval_Phase/FullNotebook_(1).ipynb
```

Notebook chi nen dong vai tro launcher:

```text
1. Mount Google Drive
2. Add Training_Eval_Phase vao sys.path
3. Import train_config
4. Run train_context_stdn.run(cfg) de train + valid + held-out test
5. Run eval_context_stdn.run_eval(cfg, data_root=...) de external eval
```

Neu chay tren Colab, dam bao folder Drive co dung path:

```text
/content/drive/MyDrive/Mining_ofMasssive
```
