from . import config


def should_retrain(metrics, threshold=None):
    threshold = config.WMAPE_THRESHOLD if threshold is None else float(threshold)
    return float(metrics["wmape"]) > threshold


def degradation_reason(metrics, threshold=None):
    threshold = config.WMAPE_THRESHOLD if threshold is None else float(threshold)
    return f"WMAPE {metrics['wmape']:.4f}% is above threshold {threshold:.4f}%"
