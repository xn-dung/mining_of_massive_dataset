import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from dataset.TLCdataset import TLCdataset
from model import CNN_LSTM_Model
from utils import custom_loss, get_metrics
from tqdm.auto import tqdm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = 8
    pred_len = 1
    crop_size = 9 
    batch_size = 64 
    epochs = 15
    loss_lambda = 0.1
    
    # feature_len = 100
    # topo_len = 32 
    npy_path = "/content/drive/MyDrive/mining_dataset/data/taxi_volume_4d_tensor.npy"

    raw_data = np.load(npy_path)
    total_time_steps = raw_data.shape
    train_size = int(0.8*total_time_steps[0])
    train_raw = raw_data[:train_size]
    eval_raw = raw_data[train_size:]

    min_val = train_raw.min(axis=(0, 2, 3), keepdims=True)
    max_val = train_raw.max(axis=(0, 2, 3), keepdims=True)

    train_dataset = TLCdataset(raw_npy=train_raw, seq_len=seq_len, pred_len=pred_len, crop_size=crop_size, min_val=min_val, max_val=max_val)
    eval_dataset = TLCdataset(raw_npy=eval_raw, seq_len=seq_len, pred_len=pred_len, crop_size=crop_size, min_val=min_val, max_val=max_val)
    
    
    target_channel_train = train_raw[:, 0, :, :] 
    label_max = target_channel_train.max().item()
    label_min = target_channel_train.min().item()
    mean_label = target_channel_train.mean().item()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    
    model = CNN_LSTM_Model(seq_len=seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(len(train_loader), len(eval_loader))


    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_img, batch_y in train_loader:
            batch_img = batch_img.to(device)
            batch_y = batch_y.to(device)
            

            target = batch_y.view(-1,1)

            # batch_x = torch.rand(batch_img.size(0), seq_len, feature_len, device=device)
            # batch_topo = torch.rand(batch_img.size(0), topo_len, device=device)

            pred = model(batch_img)
            
            loss = custom_loss(pred, target, label_max, label_min, mean_label, loss_lambda)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch: {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f}")

    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch_img, batch_y in eval_loader:
            current_batch_size = batch_img.size(0)
            
            batch_img = batch_img.to(device)
            batch_y_cpu = batch_y.numpy()
            

            # batch_x = torch.rand(current_batch_size, seq_len, feature_len, device=device)
            # batch_topo = torch.rand(current_batch_size, topo_len, device=device)
            
            batch_pred = model(batch_img).cpu().numpy()
            
            all_preds.append(batch_pred)
            all_trues.append(batch_y_cpu)

    pred = np.concatenate(all_preds, axis=0)
    true = np.concatenate(all_trues, axis=0)
    true_center = true.reshape(-1,1)
    pred_center = pred.reshape(-1,1)

    mape, rmse, mape_variant = get_metrics(true_center, pred_center, label_max, label_min)
    torch.save(model.state_dict(), f="/content/drive/MyDrive/mining_dataset/model_best.pth")
    print(f"MAPE:{mape:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE_variant: {mape_variant:.4f}")

if __name__ == '__main__':
    main()