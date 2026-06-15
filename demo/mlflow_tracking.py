from contextlib import contextmanager

from . import config


@contextmanager
def optional_mlflow_run(run_name=None):
    try:
        import mlflow
    except ImportError:
        yield None
        return

    mlflow.set_tracking_uri(str(config.MLFLOW_DIR))
    with mlflow.start_run(run_name=run_name) as run:
        yield mlflow


def log_metrics(run_name, metrics, params=None):
    with optional_mlflow_run(run_name) as mlflow:
        if mlflow is None:
            return False
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics({key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))})
        return True
