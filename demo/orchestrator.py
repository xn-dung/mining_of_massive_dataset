import json
import shutil
from pathlib import Path

import pandas as pd

from . import config, database
from .crawler import crawl_month_to_raw_dir
from .evaluate import evaluate_prediction_frame, save_metrics
from .model_adapters import stdn_adapter
from .monitor import degradation_reason, should_retrain
from .preprocessing.prepare_stdn_data import prepare_stdn_data
from .promote import compare_and_promote
from .retrain import retrain_from_checkpoint
from .utils.zone_lookup import enrich_prediction_frame
from .utils.date_utils import (
    add_months,
    days_between,
    evaluation_window,
    format_date,
    month_end_date,
    month_start_date,
    previous_month_period,
    prediction_period,
    window_label,
)
from .utils.file_utils import ensure_dir


def _metric_path(period, model_version):
    safe_period = str(period).replace(":", "-").replace("/", "-")
    return config.METRIC_DIR / f"{safe_period}_{model_version}_metrics.json"


def _prediction_path(period, model_version):
    safe_period = str(period).replace(":", "-").replace("/", "-")
    return config.PREDICTION_DIR / f"{safe_period}_{model_version}_prediction.csv"


def _existing_prediction(period, model_version):
    prediction = database.get_prediction_for_period(period, model_version)
    if prediction:
        return prediction

    path = _prediction_path(period, model_version)
    if path.exists():
        database.log_prediction(period, model_version, path, status="created")
        return {
            "period": period,
            "model_version": model_version,
            "prediction_path": str(path),
            "status": "created",
        }
    return None


def _preview_prediction(path, n=1000):
    path = Path(path)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    df = enrich_prediction_frame(df)
    return df.head(n).to_dict(orient="records")


def _require_active_model():
    active = database.get_active_model()
    if not active:
        raise FileNotFoundError(
            "No active model is registered inside demo/artifacts/models. "
            "Upload a .pth checkpoint in the dashboard before running the flow."
        )
    return active


def _period_bounds(period):
    return (
        f"{month_start_date(period)} 00:00:00",
        f"{month_end_date(period)} 23:30:00",
    )


def _prepare_crawled_data(kind, selected_date, split_train=False):
    period = previous_month_period(selected_date)
    processed_dir = config.PROCESSED_DIR / kind / period
    start_time, end_time = _period_bounds(period)

    raw_dir = crawl_month_to_raw_dir(period)
    processed = prepare_stdn_data(
        raw_dir,
        processed_dir,
        split_train=split_train,
        start_time=start_time,
        end_time=end_time,
    )
    database.log_ingested_month(period, raw_dir, processed, source_kind=f"crawl_{kind}")
    return processed


def _prepare_initial_history_data():
    data_config = config.INITIAL_HISTORY_PROCESSED_DIR / "data.json"
    if data_config.exists():
        with data_config.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        start_time = str(payload.get("start_time", ""))
        if start_time.startswith(f"{config.INITIAL_LAST_UPDATE_DATE[:4]}-{config.INITIAL_HISTORY_MONTH}-"):
            database.update_ingested_processed_dir(config.INITIAL_HISTORY_PERIOD, config.INITIAL_HISTORY_PROCESSED_DIR)
            return config.INITIAL_HISTORY_PROCESSED_DIR
        import shutil

        shutil.rmtree(config.INITIAL_HISTORY_PROCESSED_DIR, ignore_errors=True)
    if not any(config.INITIAL_HISTORY_DIR.glob("*_volume.csv")) or not any(config.INITIAL_HISTORY_DIR.glob("*_flow.parquet")):
        raise FileNotFoundError(
            "Initial history data is missing in demo/data/raw/initial_history. "
            "Seed it from Data_V_F/Raw or crawl the required month with CrawlSTDN."
        )
    processed = prepare_stdn_data(
        config.INITIAL_HISTORY_DIR,
        config.INITIAL_HISTORY_PROCESSED_DIR,
        split_train=False,
        start_time=month_start_date(config.INITIAL_HISTORY_PERIOD),
        end_time=f"{month_end_date(config.INITIAL_HISTORY_PERIOD)} 23:30:00",
    )
    database.update_ingested_processed_dir(config.INITIAL_HISTORY_PERIOD, processed)
    return processed


def _copy_month_raw_files(months, target_dir):
    target_dir = ensure_dir(target_dir)
    for path in target_dir.iterdir():
        if path.is_file():
            path.unlink()

    copied = []
    allowed_suffixes = {".csv", ".parquet", ".pq", ".npz"}
    allowed_keywords = ("volume", "flow", "context", "weather", "holiday")
    for month in months:
        period = month["period"]
        raw_dir = Path(month["raw_dir"])
        if not raw_dir.exists():
            raise FileNotFoundError(f"Ingested raw dir no longer exists for {period}: {raw_dir}")
        for path in raw_dir.iterdir():
            name = path.name.lower()
            if path.is_file() and path.suffix.lower() in allowed_suffixes and any(key in name for key in allowed_keywords):
                dst = target_dir / f"{period}_{path.name}"
                shutil.copy2(path, dst)
                copied.append(dst)
    if not copied:
        raise FileNotFoundError("No raw monthly files were found in the ingested month registry.")
    return target_dir


def _prepare_recent_ingested_data(selected_date, count=3, split_train=False):
    end_period = previous_month_period(selected_date)
    expected_periods = [add_months(end_period, offset) for offset in range(-(count - 1), 1)]
    months = [database.get_ingested_month(period) for period in expected_periods]
    missing = [period for period, month in zip(expected_periods, months) if month is None]
    if missing:
        found = ", ".join(month["period"] for month in months if month) or "none"
        raise FileNotFoundError(
            f"Need contiguous ingested months {', '.join(expected_periods)} for quarterly retrain/eval. "
            f"Missing: {', '.join(missing)}. Found: {found}. "
            "Run the monthly crawl flow for the missing months first."
        )

    period_label = f"{months[0]['period']}_to_{months[-1]['period']}"
    combined_dir = config.CRAWLED_DIR / "quarterly_combined" / period_label
    processed_dir = config.PROCESSED_DIR / "quarterly_train" / period_label
    _copy_month_raw_files(months, combined_dir)
    return prepare_stdn_data(
        combined_dir,
        processed_dir,
        split_train=split_train,
        start_time=f"{month_start_date(months[0]['period'])} 00:00:00",
        end_time=f"{month_end_date(months[-1]['period'])} 23:30:00",
    )


def _load_prediction_result(prediction):
    return {
        "case": "case_1_load_existing_prediction",
        "status": "loaded_existing_prediction",
        "message": "Prediction for this period already exists. Loading local artifact.",
        "prediction_path": prediction["prediction_path"],
        "prediction_preview": _preview_prediction(prediction["prediction_path"]),
        "state": database.get_system_state(),
    }


def load_existing_prediction_flow(selected_date):
    active = _require_active_model()
    period = prediction_period(selected_date)
    prediction = _existing_prediction(period, active["model_version"])
    if prediction:
        return _load_prediction_result(prediction)

    raise FileNotFoundError(
        f"No local prediction artifact found for {period}. "
        "Upload the latest month data so the active model can create it."
    )


def run_monthly_prediction_flow(selected_date, message=None):
    active = _require_active_model()
    period = prediction_period(selected_date)
    existing = _existing_prediction(period, active["model_version"])
    if existing:
        return _load_prediction_result(existing)

    processed_data = _prepare_crawled_data("monthly", selected_date, split_train=False)
    prediction_path = _prediction_path(period, active["model_version"])

    result = stdn_adapter.predict_period(
        active["model_path"],
        processed_data,
        prediction_path,
        model_version=active["model_version"],
    )
    metric_path = _metric_path(period, active["model_version"])
    save_metrics(result["metrics"], metric_path)

    database.log_prediction(period, active["model_version"], prediction_path)
    database.log_metrics(period, active["model_version"], result["metrics"], metric_path)
    database.update_prediction_date(selected_date)

    return {
        "case": "case_2_monthly_prediction",
        "status": "prediction_created",
        "message": message or "New monthly prediction was created.",
        "processed_data_path": str(processed_data),
        "prediction_path": str(prediction_path),
        "metric_path": str(metric_path),
        "metrics": result["metrics"],
        "prediction_preview": _preview_prediction(prediction_path),
        "state": database.get_system_state(),
    }


def run_initial_history_prediction_flow(selected_date):
    active = _require_active_model()
    period = prediction_period(selected_date)
    existing = _existing_prediction(period, active["model_version"])
    if existing:
        return _load_prediction_result(existing)

    processed_data = _prepare_initial_history_data()
    prediction_path = _prediction_path(period, active["model_version"])

    result = stdn_adapter.predict_period(
        active["model_path"],
        processed_data,
        prediction_path,
        model_version=active["model_version"],
    )
    metric_path = _metric_path(period, active["model_version"])
    save_metrics(result["metrics"], metric_path)

    database.log_prediction(period, active["model_version"], prediction_path)
    database.log_metrics(period, active["model_version"], result["metrics"], metric_path)
    database.update_prediction_date(selected_date)

    return {
        "case": "case_initial_history_prediction",
        "status": "prediction_created",
        "message": "Created prediction from seeded history data because selected date is within 30 days of last update.",
        "processed_data_path": str(processed_data),
        "prediction_path": str(prediction_path),
        "metric_path": str(metric_path),
        "metrics": result["metrics"],
        "prediction_preview": _preview_prediction(prediction_path),
        "state": database.get_system_state(),
    }


def _evaluate_active_model(selected_date, processed_data):
    active = _require_active_model()
    start_date, end_date = evaluation_window(selected_date, config.EVALUATION_WINDOW_DAYS)
    period = window_label(start_date, end_date)
    result = stdn_adapter.evaluate_model(
        active["model_path"],
        processed_data,
        actual_path=None,
        model_version=active["model_version"],
    )
    result["metrics"] = evaluate_prediction_frame(result["prediction_path"], start_date=start_date, end_date=end_date)
    metric_path = _metric_path(period, active["model_version"])
    save_metrics(result["metrics"], metric_path)
    database.log_metrics(period, active["model_version"], result["metrics"], metric_path)
    database.log_prediction(period, active["model_version"], result["prediction_path"], status="evaluated")
    return period, active, result, metric_path


def _candidate_train_data(processed_data, selected_date):
    return _prepare_recent_ingested_data(selected_date, count=3, split_train=True)


def run_evaluation_and_retrain_flow(selected_date):
    processed_data = _prepare_crawled_data("quarterly", selected_date, split_train=False)
    period, active, active_eval, active_metric_path = _evaluate_active_model(selected_date, processed_data)
    threshold = float(database.get_state("wmape_threshold", config.WMAPE_THRESHOLD))

    if not should_retrain(active_eval["metrics"], threshold=threshold):
        database.update_evaluation_date(selected_date)
        database.update_last_update_date(selected_date)
        return {
            "case": "case_3a_evaluate_keep_model",
            "status": "healthy",
            "message": "Quarterly evaluation finished. Model is within threshold; no retrain needed.",
            "processed_data_path": str(processed_data),
            "metrics": active_eval["metrics"],
            "metric_path": str(active_metric_path),
            "prediction_path": str(active_eval["prediction_path"]),
            "prediction_preview": _preview_prediction(active_eval["prediction_path"]),
            "state": database.get_system_state(),
        }

    train_data = _candidate_train_data(processed_data, selected_date)
    candidate_version, candidate_path = retrain_from_checkpoint(
        active["model_path"],
        train_data,
        base_model_version=active["model_version"],
    )
    candidate_eval = stdn_adapter.evaluate_model(
        candidate_path,
        processed_data,
        actual_path=None,
        model_version=candidate_version,
    )
    start_date, end_date = evaluation_window(selected_date, config.EVALUATION_WINDOW_DAYS)
    candidate_eval["metrics"] = evaluate_prediction_frame(
        candidate_eval["prediction_path"],
        start_date=start_date,
        end_date=end_date,
    )
    candidate_metric_path = _metric_path(period, candidate_version)
    save_metrics(candidate_eval["metrics"], candidate_metric_path)
    database.log_metrics(period, candidate_version, candidate_eval["metrics"], candidate_metric_path)
    database.log_prediction(period, candidate_version, candidate_eval["prediction_path"], status="evaluated")

    promote_result = compare_and_promote(
        active_eval["metrics"],
        candidate_eval["metrics"],
        candidate_version,
        candidate_path,
    )
    database.log_retrain_event(
        old_model_version=active["model_version"],
        candidate_model_version=candidate_version,
        promoted_model_version=candidate_version if promote_result["promoted"] else None,
        reason=degradation_reason(active_eval["metrics"], threshold),
        old_wmape=active_eval["metrics"]["wmape"],
        candidate_wmape=candidate_eval["metrics"]["wmape"],
        status=promote_result["status"],
    )
    database.update_evaluation_date(selected_date)
    database.update_last_update_date(selected_date)
    if promote_result["promoted"]:
        database.update_retrain_date(selected_date)
    else:
        database.mark_model_status(candidate_version, "archived")

    return {
        "case": "case_3b_or_3c_retrain_compare",
        "status": promote_result["status"],
        "message": (
            "Candidate model is better. Promoted to active model."
            if promote_result["promoted"]
            else "Candidate model is not better. Keeping current active model."
        ),
        "processed_data_path": str(processed_data),
        "train_data_path": str(train_data),
        "active_metrics": active_eval["metrics"],
        "candidate_metrics": candidate_eval["metrics"],
        "active_metric_path": str(active_metric_path),
        "candidate_metric_path": str(candidate_metric_path),
        "active_prediction_path": str(active_eval["prediction_path"]),
        "candidate_prediction_path": str(candidate_eval["prediction_path"]),
        "prediction_preview": _preview_prediction(candidate_eval["prediction_path"]),
        "state": database.get_system_state(),
    }


def run(selected_date):
    database.init_db()
    selected_date = format_date(selected_date)
    state = database.get_system_state()

    if days_between(state.get("last_evaluation_date"), selected_date) >= config.EVALUATION_INTERVAL_DAYS:
        return run_evaluation_and_retrain_flow(selected_date)

    active = _require_active_model()
    period = prediction_period(selected_date)
    existing = _existing_prediction(period, active["model_version"])
    if existing:
        return _load_prediction_result(existing)

    days_from_update = days_between(state.get("last_update_date"), selected_date)
    if 0 <= days_from_update <= config.PREDICTION_REFRESH_DAYS:
        return run_initial_history_prediction_flow(selected_date)

    if days_from_update > config.PREDICTION_REFRESH_DAYS:
        return run_monthly_prediction_flow(selected_date)

    return load_existing_prediction_flow(selected_date)
