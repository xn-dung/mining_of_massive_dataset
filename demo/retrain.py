from pathlib import Path

from . import config, database
from .model_adapters.stdn_adapter import train_model


def retrain_from_checkpoint(active_model_path, train_data_path, base_model_version=None):
    new_version = database.next_model_version()
    candidate_path = config.MODEL_DIR / f"{new_version}.pth"
    database.upsert_model(new_version, candidate_path, "candidate", base_model_version=base_model_version)

    trained_path = train_model(
        train_data_path=Path(train_data_path),
        base_checkpoint=Path(active_model_path),
        model_version=new_version,
        output_model_path=candidate_path,
    )
    database.upsert_model(new_version, trained_path, "candidate", base_model_version=base_model_version)
    return new_version, Path(trained_path)
