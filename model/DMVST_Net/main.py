import os
import sys
from pathlib import Path
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from dataset.TLCdataset import TLCdataset
from model import CNN_LSTM_Model
from utils import custom_loss, get_metrics
from tqdm.auto import tqdm

def plot_prediction_trends(y_true, y_pred, label_max, label_min, epoch, num_samples=200, output_path=None):
    max_value = np.array(label_max).reshape(1, -1)
    min_value = np.array(label_min).reshape(1, -1)

    y_true = y_true * (max_value - min_value) + min_value
    y_pred = y_pred * (max_value - min_value) + min_value

    y_true_sample = y_true[:num_samples]
    y_pred_sample = y_pred[:num_samples]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    axes[0].plot(y_true_sample[:, 0], label='Thực tế (True)', color='#1f77b4', linestyle='-', linewidth=2, marker='.')
    axes[0].plot(y_pred_sample[:, 0], label='Dự đoán (Predict)', color='#ff7f0e', linestyle='--', linewidth=2, marker='.')
    axes[0].set_title(f'Xu hướng Inflow (Start Volume) - {num_samples} điểm', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Số lượng xe', fontsize=12)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, linestyle=':', alpha=0.7)

    axes[1].plot(y_true_sample[:, 1], label='Thực tế (True)', color='#2ca02c', linestyle='-', linewidth=2, marker='.')
    axes[1].plot(y_pred_sample[:, 1], label='Dự đoán (Predict)', color='#d62728', linestyle='--', linewidth=2, marker='.')
    axes[1].set_title(f'Xu hướng Outflow (End Volume) - {num_samples} điểm', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Số lượng xe', fontsize=12)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Da luu bieu do tai: {output_path}")
    else:
        plt.show()
    plt.close(fig)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_year = 2024
    eval_year = 2025
    seq_len = 8
    pred_len = 1
    crop_size = 9 
    batch_size = 64 
    epochs = 20
    loss_lambda = 1
    
    feature_len = 11
    topo_len = 32 
    data_root = Path(root_path) / "data"
    train_data_dir = data_root / str(train_year)
    eval_data_dir = data_root / str(eval_year)
    train_npy_path = train_data_dir / "taxi_volume_4d_tensor.npy"
    eval_npy_path = eval_data_dir / "taxi_volume_4d_tensor.npy"
    topo_npy_path = train_data_dir / "topo_input.npy"
    train_context_npy_path = train_data_dir / "context_input.npy"
    eval_context_npy_path = eval_data_dir / "context_input.npy"

    topo_input = torch.tensor(np.load(topo_npy_path), dtype=torch.float32)
    train_context = torch.tensor(np.load(train_context_npy_path), dtype=torch.float32)
    eval_context = torch.tensor(np.load(eval_context_npy_path), dtype=torch.float32)

    train_raw = np.load(train_npy_path)
    eval_raw = np.load(eval_npy_path)

    if train_raw.shape[1:] != eval_raw.shape[1:]:
        raise ValueError(f"Train/eval tensor shape mismatch: {train_raw.shape} vs {eval_raw.shape}")
    if len(train_context) != train_raw.shape[0]:
        raise ValueError(f"Train context length {len(train_context)} does not match train tensor length {train_raw.shape[0]}")
    if len(eval_context) != eval_raw.shape[0]:
        raise ValueError(f"Eval context length {len(eval_context)} does not match eval tensor length {eval_raw.shape[0]}")

    min_val = train_raw.min(axis=(0, 2, 3), keepdims=True)
    max_val = train_raw.max(axis=(0, 2, 3), keepdims=True)

    train_dataset = TLCdataset(raw_npy=train_raw,topo_data=topo_input,context_data=train_context, seq_len=seq_len, pred_len=pred_len, crop_size=crop_size, min_val=min_val, max_val=max_val)
    eval_dataset = TLCdataset(raw_npy=eval_raw,topo_data=topo_input,context_data=eval_context, seq_len=seq_len, pred_len=pred_len, crop_size=crop_size, min_val=min_val, max_val=max_val)
    
    target_channel_train = train_raw[:, 0:2, :, :] 
    label_max = max_val[0, 0:2, 0, 0] 
    label_min = min_val[0, 0:2, 0, 0]
    mean_label = target_channel_train.mean()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    model = CNN_LSTM_Model(seq_len=seq_len,topo_len=topo_len,feature_len=feature_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4,weight_decay=1e-5)
    print(f"Train year: {train_year}, Eval year: {eval_year}")
    print(f"Số batch train: {len(train_loader)}, Số batch eval: {len(eval_loader)}")
    
    best_mape = float('inf')
    patience = 10 
    epochs_no_improve = 0
    
    best_true_center = None
    best_pred_center = None

    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_img, batch_y, batch_topo, batch_context in train_loader:
            batch_img = batch_img.to(device)
            batch_y = batch_y.to(device)
            batch_topo = batch_topo.to(device)
            batch_context = batch_context.to(device)
            
            target = batch_y.view(-1, 2)
            pred = model(batch_img, batch_topo, batch_context) 
            loss = custom_loss(pred, target, label_max, label_min, mean_label, loss_lambda)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        model.eval()
        all_preds = []
        all_trues = []

        with torch.no_grad():
            for batch_img, batch_y, batch_topo, batch_context in eval_loader:
                batch_img = batch_img.to(device)
                batch_y_cpu = batch_y.numpy()
                batch_topo = batch_topo.to(device)
                batch_context = batch_context.to(device)
                
                batch_pred = model(batch_img, batch_topo, batch_context).cpu().numpy()
                
                all_preds.append(batch_pred)
                all_trues.append(batch_y_cpu)

        pred = np.concatenate(all_preds, axis=0)
        true = np.concatenate(all_trues, axis=0)
        true_center = true.reshape(-1, 2)
        pred_center = pred.reshape(-1, 2)

        mape, rmse, mape_variant = get_metrics(true_center, pred_center, label_max, label_min)
        print(f"Epoch: {epoch+1:02d}/{epochs} | Train Loss: {avg_loss:.4f} | Eval MAPE: {mape:.4f}")

        if mape < best_mape:
            best_mape = mape
            epochs_no_improve = 0
            
            best_true_center = true_center.copy()
            best_pred_center = pred_center.copy()
            
            torch.save(model.state_dict(), str(Path(root_path) / "model_best.pth"))
            print(f"Đã lưu model tốt nhất tại epoch {epoch+1} (MAPE: {best_mape:.4f})")
            
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping.")
                break 

    print("\n--- KẾT QUẢ CUỐI CÙNG ---")
    print(f"MAPE tốt nhất đạt được: {best_mape:.4f}")
    
    if best_true_center is not None and best_pred_center is not None:
        plot_path = Path(root_path) / "artifacts" / "DMVST_Net" / "best_prediction_trends.png"
        plot_prediction_trends(
            y_true=best_true_center,
            y_pred=best_pred_center,
            label_max=label_max,
            label_min=label_min,
            epoch="Best",
            num_samples=300,
            output_path=plot_path,
        )

if __name__ == '__main__':
    main()
