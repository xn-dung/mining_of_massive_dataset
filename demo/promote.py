from pathlib import Path

from . import database


def promote_candidate(candidate_version, candidate_path):
    active = database.get_active_model()
    if active:
        database.mark_model_status(active["model_version"], "archived")
    database.set_active_model(candidate_version, Path(candidate_path))
    return {
        "promoted": True,
        "active_model_version": candidate_version,
        "active_model_path": str(candidate_path),
    }


def compare_and_promote(active_metrics, candidate_metrics, candidate_version, candidate_path):
    old_wmape = float(active_metrics["wmape"])
    candidate_wmape = float(candidate_metrics["wmape"])
    if candidate_wmape < old_wmape:
        result = promote_candidate(candidate_version, candidate_path)
        result["status"] = "promoted"
        return result
    return {
        "promoted": False,
        "status": "rejected",
        "reason": "Candidate WMAPE is not better than active model",
    }
