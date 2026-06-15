import sqlite3
import shutil
from datetime import datetime
from pathlib import Path

from . import config
from .utils.date_utils import format_date


def utc_now():
    return datetime.utcnow().isoformat(timespec="seconds")


def connect():
    config.ensure_demo_dirs()
    conn = sqlite3.connect(config.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS models (
                model_version TEXT PRIMARY KEY,
                model_path TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                base_model_version TEXT
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period TEXT NOT NULL,
                model_version TEXT NOT NULL,
                prediction_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period TEXT NOT NULL,
                model_version TEXT NOT NULL,
                wmape REAL,
                mape REAL,
                rmse REAL,
                mae REAL,
                metric_path TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS retrain_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                triggered_at TEXT NOT NULL,
                old_model_version TEXT NOT NULL,
                candidate_model_version TEXT NOT NULL,
                promoted_model_version TEXT,
                reason TEXT NOT NULL,
                old_wmape REAL,
                candidate_wmape REAL,
                status TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS ingested_months (
                period TEXT PRIMARY KEY,
                raw_dir TEXT NOT NULL,
                processed_dir TEXT,
                source_kind TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.commit()
    ensure_initial_state()


def row_to_dict(row):
    return dict(row) if row is not None else None


def get_state(key, default=None):
    with connect() as conn:
        row = conn.execute("SELECT value FROM system_state WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_state(key, value):
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO system_state(key, value)
            VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, None if value is None else str(value)),
        )
        conn.commit()


def get_system_state():
    keys = (
        "active_model_version",
        "active_model_path",
        "last_update_date",
        "last_prediction_date",
        "last_evaluation_date",
        "last_retrain_date",
        "wmape_threshold",
    )
    return {key: get_state(key) for key in keys}


def upsert_model(model_version, model_path, status, base_model_version=None):
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO models(model_version, model_path, status, created_at, base_model_version)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(model_version) DO UPDATE SET
                model_path = excluded.model_path,
                status = excluded.status,
                base_model_version = excluded.base_model_version
            """,
            (model_version, str(model_path), status, utc_now(), base_model_version),
        )
        conn.commit()


def mark_model_status(model_version, status):
    with connect() as conn:
        conn.execute("UPDATE models SET status = ? WHERE model_version = ?", (status, model_version))
        conn.commit()


def set_active_model(model_version, model_path):
    upsert_model(model_version, model_path, "active")
    set_state("active_model_version", model_version)
    set_state("active_model_path", model_path)
    config.ACTIVE_MODEL_FILE.write_text(model_version, encoding="utf-8")


def get_active_model():
    version = get_state("active_model_version")
    path = get_state("active_model_path")
    if version and path and Path(path).exists():
        return {"model_version": version, "model_path": path}
    return None


def list_models():
    with connect() as conn:
        rows = conn.execute("SELECT * FROM models ORDER BY created_at DESC").fetchall()
    return [row_to_dict(row) for row in rows]


def log_prediction(period, model_version, prediction_path, status="created"):
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO predictions(period, model_version, prediction_path, created_at, status)
            VALUES(?, ?, ?, ?, ?)
            """,
            (period, model_version, str(prediction_path), utc_now(), status),
        )
        conn.commit()


def get_latest_prediction():
    with connect() as conn:
        row = conn.execute("SELECT * FROM predictions ORDER BY created_at DESC LIMIT 1").fetchone()
    return row_to_dict(row)


def get_prediction_for_period(period, model_version=None):
    query = "SELECT * FROM predictions WHERE period = ?"
    params = [period]
    if model_version:
        query += " AND model_version = ?"
        params.append(model_version)
    query += " ORDER BY created_at DESC LIMIT 1"
    with connect() as conn:
        row = conn.execute(query, params).fetchone()
    result = row_to_dict(row)
    if result and Path(result["prediction_path"]).exists():
        return result
    return None


def list_predictions(limit=20):
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [row_to_dict(row) for row in rows]


def log_ingested_month(period, raw_dir, processed_dir=None, source_kind="uploaded"):
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO ingested_months(period, raw_dir, processed_dir, source_kind, created_at)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(period) DO UPDATE SET
                raw_dir = excluded.raw_dir,
                processed_dir = COALESCE(excluded.processed_dir, ingested_months.processed_dir),
                source_kind = excluded.source_kind
            """,
            (period, str(raw_dir), str(processed_dir) if processed_dir else None, source_kind, utc_now()),
        )
        conn.commit()


def update_ingested_processed_dir(period, processed_dir):
    with connect() as conn:
        conn.execute(
            "UPDATE ingested_months SET processed_dir = ? WHERE period = ?",
            (str(processed_dir), period),
        )
        conn.commit()


def get_ingested_month(period):
    with connect() as conn:
        row = conn.execute("SELECT * FROM ingested_months WHERE period = ?", (period,)).fetchone()
    return row_to_dict(row)


def list_ingested_months(limit=20):
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM ingested_months ORDER BY period DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [row_to_dict(row) for row in rows]


def get_recent_ingested_months(end_period, count):
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM ingested_months
            WHERE period <= ?
            ORDER BY period DESC
            LIMIT ?
            """,
            (end_period, count),
        ).fetchall()
    months = [row_to_dict(row) for row in rows]
    months.reverse()
    return months


def latest_existing_prediction():
    with connect() as conn:
        rows = conn.execute("SELECT * FROM predictions ORDER BY created_at DESC").fetchall()
    for row in rows:
        item = row_to_dict(row)
        if item and Path(item["prediction_path"]).exists():
            return item
    return None


def log_metrics(period, model_version, metrics, metric_path=None):
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO metrics(period, model_version, wmape, mape, rmse, mae, metric_path, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                period,
                model_version,
                metrics.get("wmape"),
                metrics.get("mape"),
                metrics.get("rmse"),
                metrics.get("mae"),
                str(metric_path) if metric_path else None,
                utc_now(),
            ),
        )
        conn.commit()


def list_metrics(limit=20):
    with connect() as conn:
        rows = conn.execute("SELECT * FROM metrics ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [row_to_dict(row) for row in rows]


def log_retrain_event(
    old_model_version,
    candidate_model_version,
    promoted_model_version,
    reason,
    old_wmape,
    candidate_wmape,
    status,
):
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO retrain_events(
                triggered_at, old_model_version, candidate_model_version,
                promoted_model_version, reason, old_wmape, candidate_wmape, status
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now(),
                old_model_version,
                candidate_model_version,
                promoted_model_version,
                reason,
                old_wmape,
                candidate_wmape,
                status,
            ),
        )
        conn.commit()


def list_retrain_events(limit=20):
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM retrain_events ORDER BY triggered_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [row_to_dict(row) for row in rows]


def next_model_version():
    with connect() as conn:
        rows = conn.execute("SELECT model_version FROM models").fetchall()
    highest = 0
    for row in rows:
        version = row["model_version"]
        if version.startswith("stdn_v"):
            try:
                highest = max(highest, int(version.replace("stdn_v", "")))
            except ValueError:
                pass
    return f"stdn_v{highest + 1}"


def ensure_initial_state():
    config.ensure_demo_dirs()
    seed_initial_history_files()
    if get_state("last_update_date") is None:
        set_state("last_update_date", config.INITIAL_LAST_UPDATE_DATE)
    if get_state("last_prediction_date") is None:
        set_state("last_prediction_date", None)
    elif latest_existing_prediction() is None:
        set_state("last_prediction_date", None)
    if get_state("last_evaluation_date") is None or get_state("last_evaluation_date") == "2025-01-31":
        set_state("last_evaluation_date", config.INITIAL_LAST_UPDATE_DATE)
    if get_state("last_retrain_date") is None:
        set_state("last_retrain_date", None)
    if get_state("wmape_threshold") is None:
        set_state("wmape_threshold", config.WMAPE_THRESHOLD)

    if get_active_model():
        return

    if config.INITIAL_MODEL_PATH.exists():
        set_active_model(config.INITIAL_MODEL_VERSION, config.INITIAL_MODEL_PATH)


def seed_initial_history_files():
    if config.INITIAL_RAW_SOURCE_DIR.exists():
        config.INITIAL_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        for path in config.INITIAL_HISTORY_DIR.glob("*_volume.csv"):
            if not path.name.startswith(f"{config.INITIAL_HISTORY_MONTH}_"):
                path.unlink()
        for path in config.INITIAL_HISTORY_DIR.glob("*_flow.parquet"):
            if not path.name.startswith(f"{config.INITIAL_HISTORY_MONTH}_"):
                path.unlink()

        has_seed = (config.INITIAL_HISTORY_DIR / f"{config.INITIAL_HISTORY_MONTH}_volume.csv").exists() and (
            config.INITIAL_HISTORY_DIR / f"{config.INITIAL_HISTORY_MONTH}_flow.parquet"
        ).exists()
        if not has_seed:
            for suffix in ("volume.csv", "flow.parquet"):
                src = config.INITIAL_RAW_SOURCE_DIR / f"{config.INITIAL_HISTORY_MONTH}_{suffix}"
                if src.exists():
                    shutil.copy2(src, config.INITIAL_HISTORY_DIR / src.name)

    if config.INITIAL_CONTEXT_SOURCE_PATH.exists():
        context_dst = config.INITIAL_HISTORY_DIR / "context_2024.npz"
        if not context_dst.exists():
            shutil.copy2(config.INITIAL_CONTEXT_SOURCE_PATH, context_dst)

    has_initial_raw = (
        (config.INITIAL_HISTORY_DIR / f"{config.INITIAL_HISTORY_MONTH}_volume.csv").exists()
        and (config.INITIAL_HISTORY_DIR / f"{config.INITIAL_HISTORY_MONTH}_flow.parquet").exists()
    )
    if has_initial_raw:
        log_ingested_month(
            config.INITIAL_HISTORY_PERIOD,
            config.INITIAL_HISTORY_DIR,
            config.INITIAL_HISTORY_PROCESSED_DIR if config.INITIAL_HISTORY_PROCESSED_DIR.exists() else None,
            source_kind="seeded_training_tail",
        )

    for src, dst in (
        (config.GRID_LOOKUP_SOURCE_PATH, config.GRID_LOOKUP_PATH),
        (config.ZONE_LOOKUP_SOURCE_PATH, config.ZONE_LOOKUP_PATH),
    ):
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def update_prediction_date(selected_date):
    set_state("last_prediction_date", format_date(selected_date))


def update_last_update_date(selected_date):
    set_state("last_update_date", format_date(selected_date))


def update_evaluation_date(selected_date):
    set_state("last_evaluation_date", format_date(selected_date))


def update_retrain_date(selected_date):
    set_state("last_retrain_date", format_date(selected_date))
