import sys
from pathlib import Path

import pandas as pd
import streamlit as st

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo import config, database, orchestrator
from demo.utils.file_utils import ensure_dir


def _save_uploaded_binary(uploaded, path):
    ensure_dir(Path(path).parent)
    with Path(path).open("wb") as f:
        f.write(uploaded.getbuffer())
    return path


def _table(rows):
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch")
    else:
        st.info("No records yet.")


def _prediction_table(rows):
    if not rows:
        st.info("No prediction preview yet.")
        return
    df = pd.DataFrame(rows)
    aliases = {
        "pred_inflow": "predicted_inflow_trips",
        "pred_outflow": "predicted_outflow_trips",
        "actual_inflow": "actual_inflow_trips",
        "actual_outflow": "actual_outflow_trips",
    }
    for src, dst in aliases.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    display_columns = [
        "time",
        "borough",
        "zone_name",
        "predicted_inflow_trips",
        "predicted_outflow_trips",
        "actual_inflow_trips",
        "actual_outflow_trips",
        "model_version",
    ]
    existing_columns = [col for col in display_columns if col in df.columns]
    df = df[existing_columns] if existing_columns else df
    st.dataframe(df, width="stretch")


def main():
    st.set_page_config(page_title="STDN MLOps Demo", layout="wide")
    database.init_db()
    state = database.get_system_state()

    st.title("STDN Traffic Forecasting MLOps Demo")
    st.caption("Monitor model -> detect degradation -> retrain candidate -> promote only if better.")

    left, right = st.columns([1, 2])
    with left:
        selected_date = st.date_input("Selected date", value=pd.Timestamp("2025-04-01").date())
        st.metric("Active model", state.get("active_model_version") or "unknown")
        st.write("Active model path:", state.get("active_model_path"))
        st.write("Last update:", state.get("last_update_date"))
        st.write("Last prediction:", state.get("last_prediction_date"))
        st.write("Last evaluation:", state.get("last_evaluation_date"))
        st.write("WMAPE threshold:", state.get("wmape_threshold"))
        st.write("Grid lookup path:", config.GRID_LOOKUP_PATH)
        st.write("Zone lookup path:", config.ZONE_LOOKUP_PATH)

        st.subheader("Demo-local setup")
        model_version = st.text_input("Model version", value=state.get("active_model_version") or config.INITIAL_MODEL_VERSION)
        model_upload = st.file_uploader("Upload active model checkpoint (.pth)", type=["pth", "pt", "ckpt"])
        if model_upload and st.button("Register active model"):
            model_path = config.MODEL_DIR / f"{model_version}.pth"
            _save_uploaded_binary(model_upload, model_path)
            database.set_active_model(model_version, model_path)
            st.success(f"Registered active model: {model_version}")
            st.rerun()

        grid_lookup_upload = st.file_uploader("Upload taxi_zone_lookup_grid.csv", type=["csv"])
        if grid_lookup_upload and st.button("Save grid lookup file"):
            _save_uploaded_binary(grid_lookup_upload, config.GRID_LOOKUP_PATH)
            st.success(f"Saved grid lookup: {config.GRID_LOOKUP_PATH}")
            st.rerun()

        zone_lookup_upload = st.file_uploader("Upload taxi_zone_lookup.csv", type=["csv"])
        if zone_lookup_upload and st.button("Save zone lookup file"):
            _save_uploaded_binary(zone_lookup_upload, config.ZONE_LOOKUP_PATH)
            st.success(f"Saved zone lookup: {config.ZONE_LOOKUP_PATH}")
            st.rerun()

    with right:
        st.subheader("CrawlSTDN data source")
        st.write("Data ingestion is crawl-only. When the selected date needs a new month, the demo calls `demo/CrawlSTDN/crawlSTDN.py`.")
        st.write("Crawler config:", config.CRAWL_STDN_CONFIG_PATH)
        st.write("Crawler output:", config.CRAWLED_DIR)
        st.write("If CrawlSTDN cannot fetch volume, flow, weather, and holiday files, the flow fails instead of falling back to upload.")

    if st.button("Run MLOps flow", type="primary"):
        with st.status("Running MLOps flow...", expanded=True) as status:
            st.write("Checking selected date and crawling required month when needed.")
            try:
                result = orchestrator.run(selected_date)
            except Exception as exc:
                status.update(label="Flow failed", state="error")
                st.exception(exc)
                return
            st.write(f"Finished: {result.get('case', 'unknown_case')}")
            status.update(label="MLOps flow completed", state="complete")

        st.success(result["message"])
        st.json({key: value for key, value in result.items() if key not in {"prediction_preview"}})
        if result.get("prediction_preview"):
            st.subheader("Prediction preview")
            _prediction_table(result["prediction_preview"])

    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.subheader("Models")
        _table(database.list_models())
    with col2:
        st.subheader("Metrics")
        _table(database.list_metrics())
    with col3:
        st.subheader("Retrain events")
        _table(database.list_retrain_events())
    with col4:
        st.subheader("Ingested months")
        _table(database.list_ingested_months())

    st.subheader("Predictions")
    _table(database.list_predictions())

    st.caption(f"Artifacts: {config.ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
