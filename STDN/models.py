import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention, SimpleAttention


class STDN(nn.Module):
    """
    Spatial-Temporal Dynamic Network (STDN) in PyTorch
    """
    def __init__(self, att_lstm_num, att_lstm_seq_len, lstm_seq_len, feature_vec_len, 
                 cnn_flat_size=128, lstm_out_size=128, nbhd_size=3, nbhd_type=2, 
                 flow_type=4, output_shape=2):
        super(STDN, self).__init__()
        
        self.att_lstm_num = att_lstm_num
        self.att_lstm_seq_len = att_lstm_seq_len
        self.lstm_seq_len = lstm_seq_len
        self.feature_vec_len = feature_vec_len
        self.cnn_flat_size = cnn_flat_size
        self.lstm_out_size = lstm_out_size
        self.nbhd_size = nbhd_size
        self.nbhd_type = nbhd_type
        self.flow_type = flow_type
        self.output_shape = output_shape
        
        # ===================== SHORT-TERM PART =====================
        # 1st level gate
        self.nbhd_convs_level0 = nn.ModuleList([
            nn.Conv2d(nbhd_type, 64, kernel_size=3, padding=1) 
            for _ in range(lstm_seq_len)
        ])
        self.flow_convs_level0 = nn.ModuleList([
            nn.Conv2d(flow_type, 64, kernel_size=3, padding=1) 
            for _ in range(lstm_seq_len)
        ])
        
        # 2nd level gate
        self.nbhd_convs_level1 = nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=3, padding=1) 
            for _ in range(lstm_seq_len)
        ])
        self.flow_convs_level1 = nn.ModuleList([
            nn.Conv2d(flow_type, 64, kernel_size=3, padding=1) 
            for _ in range(lstm_seq_len)
        ])
        
        # 3rd level gate
        self.nbhd_convs_level2 = nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=3, padding=1) 
            for _ in range(lstm_seq_len)
        ])
        self.flow_convs_level2 = nn.ModuleList([
            nn.Conv2d(flow_type, 64, kernel_size=3, padding=1) 
            for _ in range(lstm_seq_len)
        ])
        
        # Dense layer for CNN output
        cnn_output_size = 64 * nbhd_size * nbhd_size
        self.nbhd_dense = nn.ModuleList([
            nn.Linear(cnn_output_size, cnn_flat_size) 
            for _ in range(lstm_seq_len)
        ])
        
        # Short-term LSTM
        self.short_term_lstm = nn.LSTM(
            input_size=feature_vec_len + cnn_flat_size,
            hidden_size=lstm_out_size,
            num_layers=1,
            batch_first=True,
            dropout=0.1
        )
        
        # ===================== ATTENTION PART =====================
        # Attention LSTM CNNs (3 attention mechanisms)
        self.att_nbhd_convs_level0 = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(nbhd_type, 64, kernel_size=3, padding=1) 
                for _ in range(att_lstm_seq_len)
            ]) for _ in range(att_lstm_num)
        ])
        self.att_flow_convs_level0 = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(flow_type, 64, kernel_size=3, padding=1) 
                for _ in range(att_lstm_seq_len)
            ]) for _ in range(att_lstm_num)
        ])
        
        # 2nd level gates for attention
        self.att_nbhd_convs_level1 = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(64, 64, kernel_size=3, padding=1) 
                for _ in range(att_lstm_seq_len)
            ]) for _ in range(att_lstm_num)
        ])
        self.att_flow_convs_level1 = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(flow_type, 64, kernel_size=3, padding=1) 
                for _ in range(att_lstm_seq_len)
            ]) for _ in range(att_lstm_num)
        ])
        
        # 3rd level gates for attention
        self.att_nbhd_convs_level2 = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(64, 64, kernel_size=3, padding=1) 
                for _ in range(att_lstm_seq_len)
            ]) for _ in range(att_lstm_num)
        ])
        self.att_flow_convs_level2 = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(flow_type, 64, kernel_size=3, padding=1) 
                for _ in range(att_lstm_seq_len)
            ]) for _ in range(att_lstm_num)
        ])
        
        # Dense layers for attention CNNs
        self.att_nbhd_dense = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(cnn_output_size, cnn_flat_size) 
                for _ in range(att_lstm_seq_len)
            ]) for _ in range(att_lstm_num)
        ])
        
        # Attention LSTMs
        self.att_lstms = nn.ModuleList([
            nn.LSTM(
                input_size=feature_vec_len + cnn_flat_size,
                hidden_size=lstm_out_size,
                num_layers=1,
                batch_first=True,
                dropout=0.1
            ) for _ in range(att_lstm_num)
        ])
        
        # Attention mechanism
        self.attention_layers = nn.ModuleList([
            Attention(method='cba') for _ in range(att_lstm_num)
        ])
        
        # High-level attention LSTM
        self.attention_lstm = nn.LSTM(
            input_size=lstm_out_size * att_lstm_num,
            hidden_size=lstm_out_size,
            num_layers=1,
            batch_first=True,
            dropout=0.1
        )
        
        # Output layer
        self.output_dense = nn.Linear(lstm_out_size * 2, output_shape)

    def forward(self, att_nbhd_inputs, att_flow_inputs, att_lstm_inputs, 
                nbhd_inputs, flow_inputs, lstm_inputs):
        """
        Forward pass
        Inputs:
            att_nbhd_inputs: list of attention neighborhood inputs (att_lstm_num * att_lstm_seq_len)
            att_flow_inputs: list of attention flow inputs (att_lstm_num * att_lstm_seq_len)
            att_lstm_inputs: list of attention LSTM feature vectors (att_lstm_num)
            nbhd_inputs: list of neighborhood inputs (lstm_seq_len)
            flow_inputs: list of flow inputs (lstm_seq_len)
            lstm_inputs: LSTM feature vector input (batch, lstm_seq_len, feature_vec_len)
        """
        batch_size = lstm_inputs.shape[0]
        
        # ===================== SHORT-TERM PROCESSING =====================
        nbhd_vecs = []
        for ts in range(self.lstm_seq_len):
            # 1st level gate
            nbhd_conv = F.relu(self.nbhd_convs_level0[ts](nbhd_inputs[ts]))
            flow_conv = F.relu(self.flow_convs_level0[ts](flow_inputs[ts]))
            flow_gate = torch.sigmoid(flow_conv)
            nbhd_conv = nbhd_conv * flow_gate
            
            # 2nd level gate
            nbhd_conv = F.relu(self.nbhd_convs_level1[ts](nbhd_conv))
            flow_conv = F.relu(self.flow_convs_level1[ts](flow_inputs[ts]))
            flow_gate = torch.sigmoid(flow_conv)
            nbhd_conv = nbhd_conv * flow_gate
            
            # 3rd level gate
            nbhd_conv = F.relu(self.nbhd_convs_level2[ts](nbhd_conv))
            flow_conv = F.relu(self.flow_convs_level2[ts](flow_inputs[ts]))
            flow_gate = torch.sigmoid(flow_conv)
            nbhd_conv = nbhd_conv * flow_gate
            
            # Dense layer
            nbhd_vec = nbhd_conv.view(batch_size, -1)
            nbhd_vec = F.relu(self.nbhd_dense[ts](nbhd_vec))
            nbhd_vecs.append(nbhd_vec)
        
        # Reshape and concatenate
        nbhd_vec = torch.stack(nbhd_vecs, dim=1)  # (batch, lstm_seq_len, cnn_flat_size)
        lstm_input = torch.cat([lstm_inputs, nbhd_vec], dim=-1)
        
        # Short-term LSTM
        lstm_output, _ = self.short_term_lstm(lstm_input)
        lstm_output = lstm_output[:, -1, :]  # Take last output (batch, lstm_out_size)
        
        # ===================== ATTENTION PROCESSING =====================
        att_lstms_outputs = []
        
        for att in range(self.att_lstm_num):
            att_nbhd_vecs = []
            for ts in range(self.att_lstm_seq_len):
                idx = att * self.att_lstm_seq_len + ts
                
                # 1st level gate
                nbhd_conv = F.relu(self.att_nbhd_convs_level0[att][ts](att_nbhd_inputs[idx]))
                flow_conv = F.relu(self.att_flow_convs_level0[att][ts](att_flow_inputs[idx]))
                flow_gate = torch.sigmoid(flow_conv)
                nbhd_conv = nbhd_conv * flow_gate
                
                # 2nd level gate
                nbhd_conv = F.relu(self.att_nbhd_convs_level1[att][ts](nbhd_conv))
                flow_conv = F.relu(self.att_flow_convs_level1[att][ts](att_flow_inputs[idx]))
                flow_gate = torch.sigmoid(flow_conv)
                nbhd_conv = nbhd_conv * flow_gate
                
                # 3rd level gate
                nbhd_conv = F.relu(self.att_nbhd_convs_level2[att][ts](nbhd_conv))
                flow_conv = F.relu(self.att_flow_convs_level2[att][ts](att_flow_inputs[idx]))
                flow_gate = torch.sigmoid(flow_conv)
                nbhd_conv = nbhd_conv * flow_gate
                
                # Dense layer
                nbhd_vec = nbhd_conv.view(batch_size, -1)
                nbhd_vec = F.relu(self.att_nbhd_dense[att][ts](nbhd_vec))
                att_nbhd_vecs.append(nbhd_vec)
            
            # Reshape and concatenate
            att_nbhd_vec = torch.stack(att_nbhd_vecs, dim=1)  # (batch, att_lstm_seq_len, cnn_flat_size)
            att_lstm_input = torch.cat([att_lstm_inputs[att], att_nbhd_vec], dim=-1)
            
            # Attention LSTM
            att_lstm_out, _ = self.att_lstms[att](att_lstm_input)
            att_lstms_outputs.append(att_lstm_out)
        
        # ===================== ATTENTION COMPARISON =====================
        # Low-level attention
        att_low_level = []
        for att in range(self.att_lstm_num):
            att_output = self.attention_layers[att]([att_lstms_outputs[att], lstm_output])
            att_low_level.append(att_output)
        
        att_low_level = torch.stack(att_low_level, dim=1)  # (batch, att_lstm_num, lstm_out_size)
        
        # High-level attention LSTM
        att_high_level, _ = self.attention_lstm(att_low_level)
        att_high_level = att_high_level[:, -1, :]  # (batch, lstm_out_size)
        
        # ===================== OUTPUT =====================
        lstm_all = torch.cat([att_high_level, lstm_output], dim=-1)
        pred_volume = torch.tanh(self.output_dense(lstm_all))
        
        return pred_volume


class models:
    """Wrapper class for model creation"""
    def __init__(self):
        pass

    def stdn(self, att_lstm_num, att_lstm_seq_len, lstm_seq_len, feature_vec_len, 
             cnn_flat_size=128, lstm_out_size=128, nbhd_size=3, nbhd_type=2, 
             flow_type=4, output_shape=2):
        """Create STDN model"""
        return STDN(att_lstm_num, att_lstm_seq_len, lstm_seq_len, feature_vec_len,
                   cnn_flat_size, lstm_out_size, nbhd_size, nbhd_type, flow_type, output_shape)
