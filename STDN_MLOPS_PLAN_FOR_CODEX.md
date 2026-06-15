# STDN MLOps Demo Plan for Codex

> Mục tiêu của file này: làm bản kế hoạch ngắn gọn để triển khai một demo MLOps cho model STDN dự đoán traffic.
> Codex nên đọc file này và xây dựng hệ thống theo từng module, tránh viết toàn bộ logic lẫn vào một file.

---

## 1. Mục tiêu hệ thống

Xây dựng một pipeline MLOps demo cho model **STDN traffic forecasting**.

Hệ thống cần làm được các việc chính:

- Train model STDN ban đầu bằng dữ liệu lịch sử.
- Lưu model version, ví dụ `stdn_v1.pth`.
- Có một active model đang được dùng để predict.
- Khi user chọn một ngày trên Streamlit, hệ thống tự quyết định:
  - load prediction cũ,
  - predict tháng mới,
  - hoặc evaluate/retrain nếu đã đến hạn 3 tháng.
- Theo dõi hiệu năng model theo thời gian.
- Nếu model dự đoán kém đi, tự động retrain từ checkpoint cũ.
- Model mới chỉ được promote nếu tốt hơn model cũ trên cùng tập evaluate.
- Lưu prediction, metric, model version và retrain event để dễ demo/debug.

Câu nhớ nhanh:

```text
Monitor model -> detect degradation -> retrain candidate -> compare -> promote if better.
```

---

## 2. Ý tưởng tổng quan

Flow chính của hệ thống:

```text
Initial train STDN
  -> Save stdn_v1
  -> Set stdn_v1 as active model
  -> User opens Streamlit and selects current date
  -> Orchestrator checks system state
  -> Case 1: show existing prediction
  -> Case 2: predict next month
  -> Case 3: evaluate latest 7 days and retrain if needed
```

Flow retrain:

```text
Evaluate active model
  -> Metric > threshold?
      No  -> keep current active model
      Yes -> load old checkpoint
             retrain using latest 3 months data
             save candidate model
             evaluate candidate on same 7-day window
             if candidate better:
                 promote candidate to active model
             else:
                 keep old active model
```

---

## 3. Các trạng thái cần lưu

Nên lưu state trong SQLite hoặc một file metadata JSON cho bản demo đơn giản.

Các biến quan trọng:

```text
active_model_version      # version model đang dùng, ví dụ stdn_v1
active_model_path         # path checkpoint active model
last_prediction_date      # lần gần nhất hệ thống tạo prediction
last_evaluation_date      # lần gần nhất hệ thống evaluate model
last_retrain_date         # lần gần nhất retrain thành công
wmape_threshold           # ngưỡng retrain, ví dụ 0.25
```

Không nên dùng chung một khái niệm “ngày update gần nhất”.
Cần tách rõ:

```text
last_prediction_date  !=  last_evaluation_date  !=  selected_date
```

---

## 4. Logic khi user chọn ngày trên Streamlit

Input chính từ UI:

```text
selected_date
```

Ví dụ:

```text
selected_date = 2025-04-01
```

Hệ thống sẽ check theo thứ tự ưu tiên sau.

---

### Case 1: Chưa quá 1 tháng từ lần predict gần nhất

Điều kiện:

```text
selected_date - last_prediction_date <= 1 month
```

Hành động:

```text
- Không predict lại.
- Load prediction đã có trong artifacts/predictions.
- Hiển thị kết quả trên Streamlit.
```

Mục tiêu:

```text
Tránh chạy predict lại nếu prediction vẫn còn mới.
```

---

### Case 2: Quá 1 tháng nhưng chưa đến hạn evaluate 3 tháng

Điều kiện:

```text
selected_date - last_prediction_date > 1 month
AND
selected_date - last_evaluation_date < 3 months
```

Hành động:

```text
- User upload dữ liệu 1 tháng vừa qua.
- Preprocess dữ liệu upload.
- Active model predict 1 tháng tiếp theo.
- Lưu prediction vào artifacts/predictions.
- Cập nhật last_prediction_date.
- Hiển thị kết quả dự đoán trên Streamlit.
```

Input tạm thời từ UI:

```text
- volume.csv hoặc volume.parquet
- flow.parquet
- weather.csv
- holiday.csv
```

Sau này nếu crawl/database ổn thì thay phần upload bằng crawl tự động.

---

### Case 3: Đã đến hạn evaluate sau 3 tháng

Điều kiện:

```text
selected_date - last_evaluation_date >= 3 months
```

Hành động:

```text
- Streamlit yêu cầu user upload dữ liệu 3 tháng gần nhất.
- Hệ thống preprocess dữ liệu.
- Active model predict lại 7 ngày gần nhất đã có actual.
- Evaluate prediction với actual label.
- Tính WMAPE / MAPE / RMSE / MAE.
- So sánh metric với threshold.
```

Evaluate window:

```text
Nếu selected_date = 2025-04-01
thì evaluate 7 ngày ngay trước selected_date:
2025-03-25 -> 2025-03-31
```

Lưu ý: đây là 7 ngày, không lấy cả 24/3 nếu không cần.

Nếu metric tốt:

```text
- Không retrain.
- Cập nhật last_evaluation_date.
- Tiếp tục dùng active model cũ.
```

Nếu metric xấu:

```text
- Trigger retrain.
- Load checkpoint active model hiện tại.
- Fine-tune bằng dữ liệu 3 tháng mới upload.
- Lưu model candidate.
- Evaluate candidate trên cùng 7-day window.
- Nếu candidate tốt hơn active model thì promote.
- Nếu candidate không tốt hơn thì giữ active model cũ.
```

---

## 5. Metric và threshold

Metric chính nên dùng:

```text
WMAPE
```

Metric phụ:

```text
MAPE
RMSE
MAE
```

Lý do:

```text
Traffic data có nhiều vùng/time slot có actual rất nhỏ hoặc bằng 0.
MAPE dễ bị phóng đại khi actual gần 0.
WMAPE ổn định hơn để làm metric chính.
```

Rule bản demo:

```python
if wmape > WMAPE_THRESHOLD:
    trigger_retrain = True
else:
    trigger_retrain = False
```

Threshold ban đầu:

```python
WMAPE_THRESHOLD = 0.25
```

Không cần update threshold động ở bản demo đầu tiên.
Nếu muốn mở rộng sau:

```python
threshold = historical_best_wmape * 1.5
```

Nhưng bản đầu tiên nên dùng threshold cố định để dễ debug.

---

## 6. Cấu trúc thư mục đề xuất

```text
system/
  app.py
  config.py
  database.py
  orchestrator.py
  monitor.py
  evaluate.py
  retrain.py
  promote.py
  mlflow_tracking.py
  notifier.py

  preprocessing/
    build_context_features.py
    prepare_stdn_data.py

  model_adapters/
    base_adapter.py
    stdn_adapter.py

  utils/
    date_utils.py
    file_utils.py
    metrics.py

data/
  raw/
  uploaded/
    monthly/
    quarterly/
  processed/
    stdn/
  incoming/
    actual/

artifacts/
  models/
    stdn_v1.pth
    active_model.txt
  predictions/
  metrics/
  mlruns/
  mlops.sqlite
```

---

## 7. Module cần xây dựng

### 7.1 `config.py`

Chứa path và config chung.

Cần có:

```python
DATA_DIR
ARTIFACT_DIR
MODEL_DIR
PREDICTION_DIR
MLFLOW_DIR
SQLITE_PATH
ACTIVE_MODEL_FILE
WMAPE_THRESHOLD
PREDICTION_REFRESH_DAYS
EVALUATION_INTERVAL_DAYS
EVALUATION_WINDOW_DAYS
```

Ví dụ:

```python
PREDICTION_REFRESH_DAYS = 30
EVALUATION_INTERVAL_DAYS = 90
EVALUATION_WINDOW_DAYS = 7
WMAPE_THRESHOLD = 0.25
```

---

### 7.2 `database.py`

Quản lý SQLite.

Các bảng nên có:

```text
models
predictions
metrics
retrain_events
system_state
```

Bảng `models`:

```text
model_version
model_path
status        # active / candidate / archived
created_at
base_model_version
```

Bảng `predictions`:

```text
period
model_version
prediction_path
created_at
status        # created / evaluated
```

Bảng `metrics`:

```text
period
model_version
wmape
mape
rmse
mae
created_at
```

Bảng `retrain_events`:

```text
triggered_at
old_model_version
candidate_model_version
promoted_model_version
reason
old_wmape
candidate_wmape
status        # promoted / rejected
```

Bảng `system_state`:

```text
key
value
```

---

### 7.3 `preprocessing/build_context_features.py`

Xử lý context data.

Input:

```text
weather.csv
holiday.csv
```

Output:

```text
context.npz
```

Nhiệm vụ:

```text
- Đọc weather.csv.
- Đọc holiday.csv.
- Ghép theo date/time.
- Encode thông tin holiday/weather nếu cần.
- Lưu context feature đúng format STDN cần.
```

Tham khảo logic cũ:

```text
MockData/build_context_featuring
```

---

### 7.4 `preprocessing/prepare_stdn_data.py`

Xử lý flow/volume/context thành input cho STDN.

Input:

```text
volume.csv hoặc volume.parquet
flow.parquet
context.npz
```

Output:

```text
Volume.npz
Flow.npz
context.npz
```

Nhiệm vụ:

```text
- Đọc raw volume/flow.
- Chuẩn hóa time index.
- Convert về tensor/grid format STDN cần.
- Ghép context.
- Lưu file processed.
```

Tham khảo logic cũ:

```text
DataPreparing/
```

---

### 7.5 `model_adapters/stdn_adapter.py`

Đây là module bọc toàn bộ logic riêng của STDN.

Cần có các hàm:

```python
def load_model(model_path):
    pass


def train_model(train_data_path, base_checkpoint=None, model_version=None):
    pass


def predict_period(model_path, processed_data_path, output_path):
    pass


def evaluate_model(model_path, processed_data_path, actual_path):
    pass
```

Nguyên tắc:

```text
Code MLOps không gọi trực tiếp file STDN gốc.
Code MLOps chỉ gọi qua adapter.
```

Lợi ích:

```text
Sau này đổi STDN sang STGCN/Graph WaveNet thì chỉ cần thêm adapter mới.
```

---

### 7.6 `predict.py` hoặc `stdn_adapter.predict_period()`

Chức năng:

```text
- Load active model.
- Load processed input: Volume.npz, Flow.npz, context.npz.
- Predict traffic demand.
- Lưu prediction ra file.
```

Output prediction nên có format chung:

```text
time
grid_x
grid_y
pred_inflow
pred_outflow
model_version
```

Lưu ý:

```text
Module predict chỉ predict, không evaluate.
```

---

### 7.7 `evaluate.py`

Chức năng:

```text
- Nhận prediction file.
- Nhận actual file.
- Merge theo time, grid_x, grid_y.
- Tính WMAPE, MAPE, RMSE, MAE.
- Trả về metrics dictionary.
```

Pseudo-code:

```python
def evaluate_prediction(prediction_path, actual_path):
    pred = read_prediction(prediction_path)
    actual = read_actual(actual_path)
    merged = merge(pred, actual, on=["time", "grid_x", "grid_y"])

    metrics = {
        "wmape": compute_wmape(merged),
        "mape": compute_mape(merged),
        "rmse": compute_rmse(merged),
        "mae": compute_mae(merged),
    }

    return metrics
```

---

### 7.8 `monitor.py`

Chức năng:

```text
- Nhận metric từ evaluate.py.
- So sánh với threshold.
- Quyết định có retrain hay không.
```

Pseudo-code:

```python
def should_retrain(metrics):
    return metrics["wmape"] > WMAPE_THRESHOLD
```

---

### 7.9 `retrain.py`

Chức năng:

```text
- Load active model checkpoint.
- Train tiếp bằng dữ liệu 3 tháng mới.
- Lưu candidate model.
- Log MLflow.
```

Input:

```text
active_model_path
latest_3_months_processed_data
new_model_version
```

Output:

```text
candidate_model_path
```

Pseudo-code:

```python
def retrain_from_checkpoint(active_model_path, train_data_path):
    new_version = generate_next_version()
    candidate_path = train_model(
        train_data_path=train_data_path,
        base_checkpoint=active_model_path,
        model_version=new_version,
    )
    return candidate_path
```

---

### 7.10 `promote.py`

Chức năng:

```text
- So sánh active model và candidate model.
- Nếu candidate tốt hơn, update active_model.txt.
- Nếu không, giữ model cũ.
```

Rule:

```python
if candidate_wmape < active_wmape:
    promote_candidate()
else:
    reject_candidate()
```

---

### 7.11 `orchestrator.py`

Đây là module điều phối chính.

Pseudo-code:

```python
def run(selected_date, uploaded_files=None):
    state = load_system_state()

    if days_between(state.last_evaluation_date, selected_date) >= 90:
        return run_evaluation_and_retrain_flow(selected_date, uploaded_files)

    if days_between(state.last_prediction_date, selected_date) > 30:
        return run_monthly_prediction_flow(selected_date, uploaded_files)

    return load_existing_prediction_flow(selected_date)
```

Trong đó:

```python
def run_monthly_prediction_flow(selected_date, uploaded_files):
    # 1. validate uploaded files
    # 2. preprocess latest month data
    # 3. load active model
    # 4. predict next month
    # 5. save prediction
    # 6. update database
    # 7. return result to UI
    pass
```

```python
def run_evaluation_and_retrain_flow(selected_date, uploaded_files):
    # 1. validate latest 3 months upload
    # 2. preprocess data
    # 3. build 7-day evaluation window
    # 4. evaluate active model
    # 5. compare metric with threshold
    # 6. if good: update last_evaluation_date and stop
    # 7. if bad: retrain candidate
    # 8. evaluate candidate on same 7-day window
    # 9. promote if candidate better
    # 10. save retrain event
    pass
```

---

## 8. Streamlit UI yêu cầu

File:

```text
system/app.py
```

UI cần có:

```text
- Date picker: selected_date
- Hiển thị active model version
- Hiển thị last_prediction_date
- Hiển thị last_evaluation_date
- Hiển thị current WMAPE threshold
- Upload area cho monthly data
- Upload area cho quarterly/evaluation data
- Button: Run MLOps flow
- Progress/status message
- Prediction preview table
- Metric table
- Retrain event table
```

Status message nên có:

```text
Case 1:
"Prediction is still fresh. Loading existing prediction."

Case 2:
"New monthly prediction is required. Please upload latest month data."

Case 3:
"Quarterly evaluation is required. Please upload latest 3 months data."

Retrain:
"Model performance is below threshold. Retraining candidate model..."

Promote:
"Candidate model is better. Promoted to active model."

Reject:
"Candidate model is not better. Keeping current active model."
```

---

## 9. File upload tạm thời cho demo

Do phần crawl/database chưa ổn, bản demo dùng upload file.

### Monthly prediction upload

Dùng cho Case 2.

Required files:

```text
volume.csv hoặc volume.parquet
flow.parquet
weather.csv
holiday.csv
```

### Quarterly evaluation/retrain upload

Dùng cho Case 3.

Required files:

```text
latest_3_months_volume.csv hoặc parquet
latest_3_months_flow.parquet
latest_3_months_weather.csv
latest_3_months_holiday.csv
actual_label_for_eval_window.csv
```

Sau này thay upload bằng:

```text
- crawl from database
- read from cloud storage
- scheduled ingestion pipeline
```

---

## 10. Versioning model

Bản demo dùng đơn giản:

```text
artifacts/models/active_model.txt
```

Nội dung file:

```text
stdn_v1
```

Model checkpoints:

```text
artifacts/models/stdn_v1.pth
artifacts/models/stdn_v2.pth
artifacts/models/stdn_v3.pth
```

Version mới:

```python
stdn_v(n+1)
```

Ví dụ:

```text
Current active: stdn_v1
Retrain candidate: stdn_v2
If promoted: active_model.txt = stdn_v2
If rejected: active_model.txt = stdn_v1
```

---

## 11. MLflow logging

Nếu có MLflow thì log:

```text
- model_version
- base_model_version
- train_period
- evaluation_window
- WMAPE
- MAPE
- RMSE
- MAE
- threshold
- promoted or rejected
```

Nếu chưa có MLflow thì hệ thống vẫn chạy bình thường.
MLflow chỉ là optional tracking.

---

## 12. Thứ tự triển khai khuyến nghị

Nên code theo thứ tự sau để không bị rối.

### Step 1: Setup folder + config

```text
- Tạo system/config.py
- Tạo artifacts/models
- Tạo artifacts/predictions
- Tạo artifacts/mlops.sqlite
```

### Step 2: Database/state

```text
- Tạo database.py
- Tạo bảng models, predictions, metrics, retrain_events, system_state
- Tạo hàm get/set active model
- Tạo hàm get/update last_prediction_date, last_evaluation_date
```

### Step 3: Adapter skeleton

```text
- Tạo model_adapters/stdn_adapter.py
- Viết các hàm placeholder trước
- Sau đó mới nối vào code STDN thật
```

### Step 4: Preprocessing

```text
- Viết build_context_features.py
- Viết prepare_stdn_data.py
- Đảm bảo output đúng Volume.npz, Flow.npz, context.npz
```

### Step 5: Predict flow

```text
- Load active model
- Load processed data
- Predict
- Save prediction
- Update database
```

### Step 6: Evaluate flow

```text
- Merge prediction với actual
- Tính WMAPE/MAPE/RMSE/MAE
- Save metrics
```

### Step 7: Monitor + retrain decision

```text
- So sánh WMAPE với threshold
- Return retrain_needed True/False
```

### Step 8: Retrain candidate

```text
- Load checkpoint active model
- Train tiếp bằng data mới
- Save candidate model
```

### Step 9: Promote logic

```text
- Evaluate candidate
- Compare candidate với active model
- Update active_model.txt nếu tốt hơn
```

### Step 10: Streamlit UI

```text
- Date picker
- Upload files
- Run orchestrator
- Show metrics/predictions/retrain events
```

---

## 13. Acceptance criteria

Hệ thống được xem là hoàn thành demo nếu làm được các case sau.

### Demo Case 1

```text
User chọn ngày vẫn trong 1 tháng từ lần predict gần nhất.
Hệ thống load prediction cũ và hiển thị.
```

### Demo Case 2

```text
User chọn ngày quá 1 tháng nhưng chưa đủ 3 tháng.
User upload data 1 tháng.
Hệ thống preprocess và predict tháng tiếp theo.
Prediction được lưu vào artifacts/predictions.
```

### Demo Case 3A

```text
User chọn ngày đã đủ 3 tháng.
User upload data 3 tháng.
Hệ thống evaluate 7 ngày gần nhất.
Metric <= threshold.
Hệ thống không retrain.
```

### Demo Case 3B

```text
User chọn ngày đã đủ 3 tháng.
Metric > threshold.
Hệ thống retrain candidate model.
Candidate tốt hơn active model.
Hệ thống promote candidate thành active model mới.
```

### Demo Case 3C

```text
User chọn ngày đã đủ 3 tháng.
Metric > threshold.
Hệ thống retrain candidate model.
Candidate không tốt hơn active model.
Hệ thống giữ active model cũ.
```

---

## 14. Lưu ý quan trọng khi implement

- Không viết toàn bộ vào `app.py`.
- `app.py` chỉ gọi `orchestrator.py`.
- `orchestrator.py` không chứa code STDN trực tiếp.
- STDN phải được bọc trong `stdn_adapter.py`.
- `predict.py` không evaluate.
- `evaluate.py` không retrain.
- `retrain.py` không tự promote.
- Promote phải là bước riêng sau khi so sánh metric.
- Bản demo đầu tiên nên dùng upload file thay vì crawl tự động.
- Threshold nên để cố định trước, chưa cần dynamic threshold.
- Dùng WMAPE làm metric chính.
- Evaluate window luôn là 7 ngày ngay trước `selected_date`.

---

## 15. Tóm tắt cho Codex

Hãy xây một MLOps demo cho STDN theo kiến trúc module.
Trọng tâm không phải cải thiện model STDN, mà là xây luồng vận hành model theo thời gian.

Luồng chính:

```text
Streamlit selected_date
  -> orchestrator checks state
  -> if prediction fresh: load old prediction
  -> elif monthly refresh needed: preprocess + predict next month
  -> elif quarterly evaluation needed: evaluate latest 7 days
       -> if metric good: keep model
       -> if metric bad: retrain candidate
            -> evaluate candidate
            -> promote only if candidate better
```

Kết quả cần có:

```text
- active model management
- prediction artifact
- metric logging
- retrain event logging
- Streamlit dashboard
- clean separation between MLOps core and STDN adapter
```
