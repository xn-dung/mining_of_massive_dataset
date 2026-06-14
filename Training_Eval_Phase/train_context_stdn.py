import torch
import torch.nn as nn
import torch.optim as optim

import train_config as cfg
from train_utils import (
    create_model,
    create_test_loader,
    create_train_val_loaders,
    evaluate,
    save_prediction_outputs,
    save_checkpoint,
    train_one_epoch,
    validate_with_metrics,
)


def run(config=cfg):
    train_loader, val_loader, shape_info, data_config, runtime_config = create_train_val_loaders(config)
    model, device = create_model(config, shape_info, data_config)
    
    checkpooint = torch.load("E:/CodeMiningCuoiKy/mining_of_massive_dataset/PathModel/stdn_context_jun_dec_2024.pth", map_location=device)
    model.load_state_dict(checkpooint)
    print(f"Loaded checkpoint for warm start: {config.MODEL_SAVE_PATH}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"Context dim: {shape_info.get('context_dim', 0)}")
    print(f"Runtime data config: {runtime_config}")

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(1, config.MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, config, epoch)
        val_metrics = validate_with_metrics(model, val_loader, criterion, data_config, device, config)
        val_loss = val_metrics["loss"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch}/{config.MAX_EPOCHS} - "
            f"train_loss: {train_loss:.6f} - "
            f"val_loss: {val_loss:.6f} - "
            f"val_RMSE: {val_metrics['rmse']:.4f} - "
            f"val_WMAPE: {val_metrics['wmape']:.2f}% - "
            f"val_Filtered_MAPE: {val_metrics['filtered_mape']:.2f}% - "
            f"val_R2: {val_metrics['r2']:.4f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                model,
                config,
                {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "shape_info": shape_info,
                    "data_config": data_config,
                },
            )
            print(f"Saved best checkpoint: {config.MODEL_SAVE_PATH}")

    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best checkpoint for held-out test: {config.MODEL_SAVE_PATH}")

    test_loader, test_shape_info = create_test_loader(config, runtime_config=runtime_config)
    metrics = evaluate(model, test_loader, data_config, device, config)
    save_prediction_outputs(metrics, config.TRAIN_TEST_PREDICTIONS_PATH, config.TRAIN_TEST_METRICS_PATH)
    print("--- Held-out Test Results ---")
    print(
        f"RMSE: {metrics['rmse']:.4f} | "
        f"WMAPE: {metrics['wmape']:.2f}% | "
        f"Filtered MAPE: {metrics['filtered_mape']:.2f}% | "
        f"R2: {metrics['r2']:.4f}"
    )
    print(f"Saved held-out test predictions: {config.TRAIN_TEST_PREDICTIONS_PATH}")

    return {
        "model": model,
        "device": device,
        "history": history,
        "metrics": metrics,
        "shape_info": shape_info,
        "test_shape_info": test_shape_info,
        "data_config": data_config,
        "runtime_config": runtime_config,
    }


if __name__ == "__main__":
    run()
