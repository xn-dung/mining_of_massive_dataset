import torch
from pathlib import Path

import train_config as cfg
from train_utils import create_model, create_test_loader_from_data_root, evaluate, save_prediction_outputs


def load_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint


def run_eval(config=cfg, checkpoint_path=None, save_outputs=True, data_root=None):
    data_root = Path(data_root or config.EVAL_DATA_ROOT)
    test_loader, test_shape_info, data_config, runtime_config = create_test_loader_from_data_root(config, data_root)
    model, device = create_model(config, test_shape_info, data_config)

    checkpoint_path = checkpoint_path or config.MODEL_SAVE_PATH
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    metrics = evaluate(model, test_loader, data_config, device, config)

    print(f"External eval data root: {data_root}")
    print(f"Runtime data config: {runtime_config}")
    print(f"Loaded checkpoint: {checkpoint_path}")
    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print("--- External Eval Results ---")
    print(
        f"RMSE: {metrics['rmse']:.4f} | "
        f"WMAPE: {metrics['wmape']:.2f}% | "
        f"Filtered MAPE: {metrics['filtered_mape']:.2f}% | "
        f"R2: {metrics['r2']:.4f}"
    )

    outputs = {}
    if save_outputs:
        pred_path, metric_path = save_prediction_outputs(
            metrics,
            config.EVAL_PREDICTIONS_PATH,
            config.EVAL_METRICS_PATH,
        )
        outputs = {"predictions_path": pred_path, "metrics_path": metric_path}
        print(f"Saved predictions: {pred_path}")
        print(f"Saved metrics: {metric_path}")

    return {
        "model": model,
        "device": device,
        "metrics": metrics,
        "test_shape_info": test_shape_info,
        "data_config": data_config,
        "runtime_config": runtime_config,
        **outputs,
    }


if __name__ == "__main__":
    run_eval()
