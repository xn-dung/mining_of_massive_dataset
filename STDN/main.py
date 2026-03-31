import sys
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import file_loader as file_loader
import models as models_module

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

parser = argparse.ArgumentParser(description='Spatial-Temporal Dynamic Network (PyTorch)')
parser.add_argument('--dataset', type=str, default='taxi', help='taxi or bike')
parser.add_argument('--batch_size', type=int, default=64, help='size of batch')
parser.add_argument('--max_epochs', type=int, default=1000, help='maximum epochs')
parser.add_argument('--att_lstm_num', type=int, default=3,
                    help='the number of time for attention (i.e., value of Q in the paper)')
parser.add_argument('--long_term_lstm_seq_len', type=int, default=3,
                    help='the number of days for attention mechanism (i.e., value of P in the paper)')
parser.add_argument('--short_term_lstm_seq_len', type=int, default=7,
                    help='the length of short term value')
parser.add_argument('--cnn_nbhd_size', type=int, default=3,
                    help='neighbors for local cnn (2*cnn_nbhd_size+1) for area size')
parser.add_argument('--nbhd_size', type=int, default=2,
                    help='for feature extraction')
parser.add_argument('--cnn_flat_size', type=int, default=128,
                    help='dimension of local conv output')
parser.add_argument('--model_name', type=str, default='stdn',
                    help='model name')
parser.add_argument('--early_stop_patience', type=int, default=5,
                    help='patience for early stopping')
parser.add_argument('--start_epoch_for_early_stop', type=int, default=40,
                    help='start epoch for early stopping')
parser.add_argument('--learning_rate', type=float, default=1e-4,
                    help='learning rate')

args = parser.parse_args()
print(args)


class EarlyStopping:
    """Early stopping to avoid overfitting"""
    def __init__(self, patience=5, verbose=False, start_epoch=40, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.start_epoch = start_epoch
        self.delta = delta

    def __call__(self, val_loss, epoch):
        if epoch < self.start_epoch:
            return
        
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True


def eval_together(y, pred_y, threshold):
    """Evaluate RMSE and MAPE"""
    mask = y > threshold
    if np.sum(mask) == 0:
        return -1, -1
    mape = np.mean(np.abs(y[mask] - pred_y[mask]) / y[mask])
    rmse = np.sqrt(np.mean(np.square(y[mask] - pred_y[mask])))
    return rmse, mape


def eval_lstm(y, pred_y, threshold):
    """Evaluate pickup and dropoff separately"""
    pickup_y = y[:, 0]
    dropoff_y = y[:, 1]
    pickup_pred_y = pred_y[:, 0]
    dropoff_pred_y = pred_y[:, 1]
    pickup_mask = pickup_y > threshold
    dropoff_mask = dropoff_y > threshold
    
    avg_pickup_rmse, avg_pickup_mape = -1, -1
    avg_dropoff_rmse, avg_dropoff_mape = -1, -1
    
    # pickup part
    if np.sum(pickup_mask) != 0:
        avg_pickup_mape = np.mean(np.abs(pickup_y[pickup_mask] - pickup_pred_y[pickup_mask]) / pickup_y[pickup_mask])
        avg_pickup_rmse = np.sqrt(np.mean(np.square(pickup_y[pickup_mask] - pickup_pred_y[pickup_mask])))
    
    # dropoff part
    if np.sum(dropoff_mask) != 0:
        avg_dropoff_mape = np.mean(np.abs(dropoff_y[dropoff_mask] - dropoff_pred_y[dropoff_mask]) / dropoff_y[dropoff_mask])
        avg_dropoff_rmse = np.sqrt(np.mean(np.square(dropoff_y[dropoff_mask] - dropoff_pred_y[dropoff_mask])))

    return (avg_pickup_rmse, avg_pickup_mape), (avg_dropoff_rmse, avg_dropoff_mape)


def collate_fn(batch):
    """Custom collate function to handle list of tensors"""
    # Unzip batch
    att_cnn, att_flow, att_lstm, cnn, flow, lstm, label = zip(*batch)
    
    # Stack CNN features (reshape for Conv2D input: batch, channels, height, width)
    att_cnn_stacked = [torch.stack([sample[i] for sample in att_cnn]) for i in range(len(att_cnn[0]))]
    att_cnn_stacked = [x.permute(0, 3, 1, 2) for x in att_cnn_stacked]  # Move channel to dim 1
    
    att_flow_stacked = [torch.stack([sample[i] for sample in att_flow]) for i in range(len(att_flow[0]))]
    att_flow_stacked = [x.permute(0, 3, 1, 2) for x in att_flow_stacked]
    
    att_lstm_stacked = torch.stack(att_lstm)
    
    cnn_stacked = [torch.stack([sample[i] for sample in cnn]) for i in range(len(cnn[0]))]
    cnn_stacked = [x.permute(0, 3, 1, 2) for x in cnn_stacked]
    
    flow_stacked = [torch.stack([sample[i] for sample in flow]) for i in range(len(flow[0]))]
    flow_stacked = [x.permute(0, 3, 1, 2) for x in flow_stacked]
    
    lstm_stacked = torch.stack(lstm)
    label_stacked = torch.stack(label)
    
    return att_cnn_stacked, att_flow_stacked, att_lstm_stacked, cnn_stacked, flow_stacked, lstm_stacked, label_stacked


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        att_cnn, att_flow, att_lstm, cnn, flow, lstm, labels = batch
        
        # Move to device
        att_cnn = [x.to(device) for x in att_cnn]
        att_flow = [x.to(device) for x in att_flow]
        att_lstm = [x.to(device) for x in att_lstm]
        cnn = [x.to(device) for x in cnn]
        flow = [x.to(device) for x in flow]
        lstm = lstm.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(att_cnn, att_flow, att_lstm, cnn, flow, lstm)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f'  Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.6f}')
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            att_cnn, att_flow, att_lstm, cnn, flow, lstm, labels = batch
            
            # Move to device
            att_cnn = [x.to(device) for x in att_cnn]
            att_flow = [x.to(device) for x in att_flow]
            att_lstm = [x.to(device) for x in att_lstm]
            cnn = [x.to(device) for x in cnn]
            flow = [x.to(device) for x in flow]
            lstm = lstm.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(att_cnn, att_flow, att_lstm, cnn, flow, lstm)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def predict(model, data_loader, device):
    """Get predictions on dataset"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in data_loader:
            att_cnn, att_flow, att_lstm, cnn, flow, lstm, _ = batch
            
            # Move to device
            att_cnn = [x.to(device) for x in att_cnn]
            att_flow = [x.to(device) for x in att_flow]
            att_lstm = [x.to(device) for x in att_lstm]
            cnn = [x.to(device) for x in cnn]
            flow = [x.to(device) for x in flow]
            lstm = lstm.to(device)
            
            # Forward pass
            outputs = model(att_cnn, att_flow, att_lstm, cnn, flow, lstm)
            predictions.append(outputs.cpu().numpy())
    
    return np.concatenate(predictions, axis=0)


class STDNDataset(torch.utils.data.Dataset):
    """Custom dataset for STDN"""
    def __init__(self, att_cnn, att_flow, att_lstm, cnn, flow, lstm, labels):
        self.att_cnn = att_cnn
        self.att_flow = att_flow
        self.att_lstm = att_lstm
        self.cnn = cnn
        self.flow = flow
        self.lstm = lstm
        self.labels = labels
        
        # Number of samples
        self.n_samples = lstm.shape[0]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Collect features for this sample across all time steps
        att_cnn_sample = [self.att_cnn[i][idx] for i in range(len(self.att_cnn))]
        att_flow_sample = [self.att_flow[i][idx] for i in range(len(self.att_flow))]
        att_lstm_sample = [self.att_lstm[i][idx] for i in range(len(self.att_lstm))]
        cnn_sample = [self.cnn[i][idx] for i in range(len(self.cnn))]
        flow_sample = [self.flow[i][idx] for i in range(len(self.flow))]
        lstm_sample = self.lstm[idx]
        label_sample = self.labels[idx]
        
        return att_cnn_sample, att_flow_sample, att_lstm_sample, cnn_sample, flow_sample, lstm_sample, label_sample


def main(batch_size=64, max_epochs=100, validation_split=0.2, early_stop_patience=5):
    """Main training function"""
    
    checkpoint_dir = "./checkpoints/"
    
    # Create checkpoint directory if not exists
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load sampler
    if args.dataset == 'taxi':
        sampler = file_loader.file_loader()
    elif args.dataset == 'bike':
        sampler = file_loader.file_loader(config_path="data_bike.json")
    else:
        raise ValueError("Can not recognize dataset, please enter taxi or bike")
    
    # Create model
    modeler = models_module.models()
    
    if args.model_name == "stdn":
        print(f"Loading {args.dataset} dataset...")
        # Training data
        att_cnnx, att_flow, att_x, cnnx, flow, x, y = sampler.sample_stdn(
            datatype="train",
            att_lstm_num=args.att_lstm_num,
            long_term_lstm_seq_len=args.long_term_lstm_seq_len,
            short_term_lstm_seq_len=args.short_term_lstm_seq_len,
            nbhd_size=args.nbhd_size,
            cnn_nbhd_size=args.cnn_nbhd_size,
            return_torch=True
        )

        print(f"Start training {args.model_name} with input shape {cnnx[0].shape} / {x.shape}")

        # Create model
        model = modeler.stdn(
            att_lstm_num=args.att_lstm_num,
            att_lstm_seq_len=args.long_term_lstm_seq_len,
            lstm_seq_len=len(cnnx),
            feature_vec_len=x.shape[-1],
            cnn_flat_size=args.cnn_flat_size,
            nbhd_size=cnnx[0].shape[1],
            nbhd_type=cnnx[0].shape[0]
        )
        
        model.to(device)
        print(f"Model moved to {device}")
        
        # Create dataset
        dataset = STDNDataset(att_cnnx, att_flow, att_x, cnnx, flow, x, y)
        
        # Split into train/val
        n_samples = len(dataset)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val]
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        early_stopping = EarlyStopping(
            patience=early_stop_patience,
            verbose=True,
            start_epoch=args.start_epoch_for_early_stop
        )
        
        # Training loop
        print("Starting training...")
        for epoch in range(max_epochs):
            print(f"\nEpoch [{epoch + 1}/{max_epochs}]")
            
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            print(f"Training Loss: {train_loss:.6f}")
            
            # Validate
            val_loss = validate(model, val_loader, criterion, device)
            print(f"Validation Loss: {val_loss:.6f}")
            
            # Early stopping
            early_stopping(val_loss, epoch)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        print("\nTraining complete. Evaluating on test set...")
        
        # Load test data
        att_cnnx_test, att_flow_test, att_x_test, cnnx_test, flow_test, x_test, y_test = \
            sampler.sample_stdn(
                datatype="test",
                att_lstm_num=args.att_lstm_num,
                long_term_lstm_seq_len=args.long_term_lstm_seq_len,
                short_term_lstm_seq_len=args.short_term_lstm_seq_len,
                nbhd_size=args.nbhd_size,
                cnn_nbhd_size=args.cnn_nbhd_size,
                return_torch=True
            )
        
        # Create test dataset
        test_dataset = STDNDataset(
            att_cnnx_test, att_flow_test, att_x_test,
            cnnx_test, flow_test, x_test, y_test
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Predict
        y_pred = predict(model, test_loader, device)
        
        # Evaluate
        threshold = float(sampler.threshold) / sampler.config["volume_train_max"]
        print("Evaluating threshold: {0}.".format(threshold))
        
        # Convert to numpy for evaluation
        y_test_np = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test
        
        (prmse, pmape), (drmse, dmape) = eval_lstm(y_test_np, y_pred, threshold)
        print(
            "Test on model {0}:\n"
            "pickup rmse = {1:.6f}, pickup mape = {2:.2f}%\n"
            "dropoff rmse = {3:.6f}, dropoff mape = {4:.2f}%".format(
                args.model_name, prmse, pmape * 100, drmse, dmape * 100
            )
        )
        
        # Save model
        curr_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        model_path = checkpoint_dir + args.model_name + "_" + curr_time + ".pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        return
    
    else:
        print("Cannot recognize parameter...")
        return


if __name__ == "__main__":
    main(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        early_stop_patience=args.early_stop_patience
    )
