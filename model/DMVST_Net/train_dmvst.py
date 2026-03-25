import torch
import numpy as np
from model import CNN_LSTM_Model
from utils import sample_get, sample_get_network, sample_get_static, custom_loss, get_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seq_len = 8
feature_len = 100
topo_len = 32
batch_size = 64
epochs = 200
loss_lambda = 10.0

datasource = np.random.rand(1000, feature_len + 1)
network_data = np.random.rand(1000, 9, 9)
static_data = np.random.rand(1000, topo_len)

cnt = 1

trainX, trainY = sample_get(datasource, seq_len, cnt)
trainimage = sample_get_network(network_data, seq_len, cnt)
traintopo = sample_get_static(static_data, seq_len, cnt)

label_max = trainY.max()
label_min = trainY.min()
mean_label = trainY.mean()

trainX = torch.tensor(trainX, dtype=torch.float32).to(device)
trainY = torch.tensor(trainY, dtype=torch.float32).to(device)
trainimage = torch.tensor(trainimage, dtype=torch.float32).unsqueeze(-1).to(device)
traintopo = torch.tensor(traintopo, dtype=torch.float32).to(device)

model = CNN_LSTM_Model(seq_len=seq_len, feature_len=feature_len, topo_len=topo_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    model.train()

    perm = torch.randperm(trainX.size(0))

    total_loss = 0

    for i in range(0, trainX.size(0), batch_size):
        idx = perm[i:i+batch_size]

        batch_x = trainX[idx]
        batch_y = trainY[idx]
        batch_img = trainimage[idx]
        batch_topo = traintopo[idx]

        pred = model(batch_img, batch_x, batch_topo)

        loss = custom_loss(pred, batch_y, label_max, label_min, mean_label, loss_lambda)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch:", epoch, "Loss:", total_loss)

model.eval()

with torch.no_grad():
    pred = model(trainimage, trainX, traintopo).cpu().numpy()
    true = trainY.cpu().numpy()

    smape, rmse, mape_variant = get_metrics(true, pred, label_max, label_min)

    print("SMAPE:", smape)
    print("RMSE:", rmse)
    print("MAPE_variant:", mape_variant)