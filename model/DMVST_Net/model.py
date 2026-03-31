import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalSeqConv(nn.Module):
    def __init__(self, seq_len, in_channels, out_channels, kernel_size=3, padding=1):
        super(LocalSeqConv, self).__init__()
        self.seq_len = seq_len
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x): #(Batch_size, Time, Channel, Height, Weight)
        B,T,C,H,W = x.shape
        x = x.reshape(B * T, C, H, W)
        out = F.relu(self.conv(x))
        out = out.view(B, T, out.shape[1], out.shape[2], out.shape[3])
        return out


class CNN_LSTM_Model(nn.Module):
    def __init__(self, seq_len=8, feature_len=100, topo_len=32):
        super(CNN_LSTM_Model, self).__init__()

        self.conv1 = LocalSeqConv(seq_len, 1, 64)
        self.bn1 = nn.BatchNorm3d(64)

        self.conv2 = LocalSeqConv(seq_len, 64, 64)
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = LocalSeqConv(seq_len, 64, 64)

        # Output của CNN là (B,T,64,9,9)

        self.fc_spatial = nn.Linear(64 * 9  * 9, 64)

        self.lstm = nn.LSTM(feature_len + 64, 512, batch_first=True)

        self.topo_fc = nn.Linear(topo_len, 6)

        self.final_fc = nn.Linear(512 + 6, 1)

    def forward(self, image, lstm_input, topo_input):

    

        x = self.conv1(image)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.bn1(x)
        x = x.permute(0, 2, 1, 3, 4) 

        x = self.conv2(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.bn2(x)
        x = x.permute(0, 2, 1, 3, 4)

        x = self.conv3(x)

        B, T, C, H, W = x.shape
        x = x.view(B, T, -1)

        spatial_out = torch.relu(self.fc_spatial(x))
        lstm_in = torch.cat([lstm_input, spatial_out], dim=-1)

        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = lstm_out[:, -1, :]

        topo_emb = torch.relu(self.topo_fc(topo_input))
        final_in = torch.cat([lstm_out, topo_emb], dim=-1)
        out = torch.sigmoid(self.final_fc(final_in))

        return out