# STDN MLOps Demo

This folder contains a modular MLOps demo for the existing STDN traffic forecasting code.

Run the dashboard from the repository root:

```powershell
streamlit run demo/app.py
```

The demo keeps runtime state in:

```text
demo/artifacts/mlops.sqlite
demo/artifacts/models/active_model.txt
demo/artifacts/predictions/
demo/artifacts/metrics/
```

Main flow:

```text
selected_date
  -> orchestrator
  -> load seeded Dec 2024 history, monthly predict, or quarterly evaluate/retrain
  -> STDN work is routed through model_adapters/stdn_adapter.py
```

Demo calendar:

```text
Initial known history: 2024-12, registered in SQLite as an ingested month.
Case 1: selected date within 30 days from 2024-12-31 uses the seeded Dec 2024 data.
Case 2: selected date after 30 days and before 90 days crawls the previous month with demo/CrawlSTDN.
Case 3: selected date after 90 days crawls the newest missing month, then combines the 3 contiguous months immediately before the selected date for eval/retrain.
```

Predictions written by `model_adapters/stdn_adapter.py` are denormalized trip counts. The preprocessing uses `Data_V_F/prepare_6-12_2024/data.json` scalers when available, so crawled demo data is normalized consistently with the original 6-12/2024 training checkpoint.

The demo data ingestion is crawl-only. It calls `demo/CrawlSTDN/crawlSTDN.py` and writes raw monthly files into `demo/data/crawled/<YYYY-MM>/`. If CrawlSTDN cannot fetch required files, the flow fails; there is no data-upload fallback.

The demo is intentionally isolated for runtime artifacts. Register these inside the dashboard:

```text
demo/artifacts/models/<model_version>.pth
demo/data/raw/taxi_zone_lookup_grid.csv
```
