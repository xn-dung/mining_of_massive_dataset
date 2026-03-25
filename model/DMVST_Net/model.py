import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalSeqConv(nn.Module):
    def __init__(self, seq_len, in_channels, out_channels, kernel_size=3, padding=1):
        super(LocalSeqConv, self).__init__()
        self.seq_len = seq_len
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            for _ in range(seq_len)
        ])

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        outputs = []
        for t in range(self.seq_len):
            out = self.convs[t](x[:, t])
            out = F.relu(out)
            outputs.append(out)
        return torch.stack(outputs, dim=1)


class CNN_LSTM_Model(nn.Module):
    def __init__(self, seq_len=8, feature_len=100, topo_len=32):
        super(CNN_LSTM_Model, self).__init__()

        self.conv1 = LocalSeqConv(seq_len, 1, 32)
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = LocalSeqConv(seq_len, 32, 32)
        self.bn2 = nn.BatchNorm3d(32)

        self.conv3 = LocalSeqConv(seq_len, 32, 32)

        self.fc_spatial = nn.Linear(32 * 9 * 9, 64)

        self.lstm = nn.LSTM(feature_len + 64, 512, batch_first=True)

        self.topo_fc = nn.Linear(topo_len, 6)

        self.final_fc = nn.Linear(512 + 6, 1)

    def forward(self, image, lstm_input, topo_input):
        x = self.conv1(image)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.conv3(x)

        B, T, C, H, W = x.shape
        x = x.view(B, T, -1)

        spatial_out = torch.relu(self.fc_spatial(x))

        x = torch.cat([lstm_input, spatial_out], dim=-1)

        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]

        topo_emb = torch.tanh(self.topo_fc(topo_input))

        x = torch.cat([lstm_out, topo_emb], dim=-1)

        out = torch.sigmoid(self.final_fc(x))

        return out