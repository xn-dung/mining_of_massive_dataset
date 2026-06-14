import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention, SimpleAttention


class STDN(nn.Module):
    def __init__(self, att_lstm_num, att_lstm_seq_len, lstm_seq_len, feature_vec_len, 
                 cnn_flat_size=128, lstm_out_size=128, nbhd_size=3, nbhd_type=2, 
                 flow_type=4, output_shape=2, dropout_rate=0.5, context_dim=0):
        super(STDN, self).__init__()
        
        self.att_lstm_num = att_lstm_num
        self.att_lstm_seq_len = att_lstm_seq_len
        self.lstm_seq_len = lstm_seq_len
        self.feature_vec_len = feature_vec_len
        self.cnn_flat_size = cnn_flat_size
        self.lstm_out_size = lstm_out_size
        self.dropout = nn.Dropout(dropout_rate)
        self.context_dim = context_dim

        if context_dim > 0:
            self.context_projection = nn.Linear(context_dim, 64)
            nn.init.zeros_(self.context_projection.weight)
            nn.init.zeros_(self.context_projection.bias)
        else:
            self.context_projection = None
        
        self.nbhd_convs_level0 = nn.ModuleList([
            nn.Conv2d(nbhd_type, 64, 3, padding=1) 
            for _ in range(lstm_seq_len)
        ])
        self.flow_convs_level0 = nn.ModuleList([
            nn.Conv2d(flow_type, 64, 3, padding=1) 
            for _ in range(lstm_seq_len)
        ])
        
        self.nbhd_convs_level1 = nn.ModuleList([
            nn.Conv2d(64, 64, 3, padding=1) 
            for _ in range(lstm_seq_len)
        ])
        self.flow_convs_level1 = nn.ModuleList([
            nn.Conv2d(flow_type, 64, 3, padding=1) 
            for _ in range(lstm_seq_len)
        ])
        
        self.nbhd_convs_level2 = nn.ModuleList([
            nn.Conv2d(64, 64, 3, padding=1) 
            for _ in range(lstm_seq_len)
        ])
        self.flow_convs_level2 = nn.ModuleList([
            nn.Conv2d(flow_type, 64, 3, padding=1) 
            for _ in range(lstm_seq_len)
        ])
        
        cnn_output_size = 64 * nbhd_size * nbhd_size
        
        self.nbhd_dense = nn.ModuleList([
            nn.Linear(cnn_output_size, cnn_flat_size) 
            for _ in range(lstm_seq_len)
        ])
        
        self.short_term_lstm = nn.LSTM(
            input_size=feature_vec_len + cnn_flat_size,
            hidden_size=lstm_out_size,
            batch_first=True
        )

        self.att_nbhd_convs_level0 = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(nbhd_type, 64, 3, padding=1) 
                for _ in range(att_lstm_seq_len)
            ]) for _ in range(att_lstm_num)
        ])
        self.att_flow_convs_level0 = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(flow_type, 64, 3, padding=1) 
                for _ in range(att_lstm_seq_len)
            ]) for _ in range(att_lstm_num)
        ])
        
        self.att_nbhd_convs_level1 = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(64, 64, 3, padding=1) 
                for _ in range(att_lstm_seq_len)
            ]) for _ in range(att_lstm_num)
        ])
        self.att_flow_convs_level1 = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(flow_type, 64, 3, padding=1) 
                for _ in range(att_lstm_seq_len)
            ]) for _ in range(att_lstm_num)
        ])
        
        self.att_nbhd_convs_level2 = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(64, 64, 3, padding=1) 
                for _ in range(att_lstm_seq_len)
            ]) for _ in range(att_lstm_num)
        ])
        self.att_flow_convs_level2 = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(flow_type, 64, 3, padding=1) 
                for _ in range(att_lstm_seq_len)
            ]) for _ in range(att_lstm_num)
        ])
        
        self.att_nbhd_dense = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(cnn_output_size, cnn_flat_size) 
                for _ in range(att_lstm_seq_len)
            ]) for _ in range(att_lstm_num)
        ])
        
        self.att_lstms = nn.ModuleList([
            nn.LSTM(feature_vec_len + cnn_flat_size, lstm_out_size, batch_first=True)
            for _ in range(att_lstm_num)
        ])
        
        self.attention_layers = nn.ModuleList([
            Attention(method='cba', att_size=lstm_out_size, query_dim=lstm_out_size)
            for _ in range(att_lstm_num)
        ])
        
        self.attention_lstm = nn.LSTM(
            input_size=lstm_out_size,
            hidden_size=lstm_out_size,
            batch_first=True
        )
        
        self.output_dense = nn.Linear(lstm_out_size * 2, output_shape)

    def _context_at(self, context_inputs, ts):
        if context_inputs is None:
            return None
        if isinstance(context_inputs, (list, tuple)):
            return context_inputs[ts]
        if context_inputs.dim() == 3:
            return context_inputs[:, ts, :]
        return context_inputs

    def _att_context_at(self, att_context_inputs, att, ts, idx):
        if att_context_inputs is None:
            return None
        if isinstance(att_context_inputs, (list, tuple)):
            item = att_context_inputs[att]
            if isinstance(item, (list, tuple)):
                return item[ts]
            if item.dim() == 3:
                return item[:, ts, :]
            return att_context_inputs[idx]
        if att_context_inputs.dim() == 4:
            return att_context_inputs[:, att, ts, :]
        if att_context_inputs.dim() == 3:
            if att_context_inputs.shape[1] == self.att_lstm_num:
                return att_context_inputs[:, att, :]
            return att_context_inputs[:, idx, :]
        return att_context_inputs

    def _add_context(self, x, context):
        if context is None:
            return x
        if self.context_projection is None:
            raise ValueError("context_dim must be > 0 when context inputs are provided.")

        if context.dim() > 2:
            context = context.reshape(context.shape[0], -1)
        context = context.to(device=x.device, dtype=x.dtype)
        external = self.context_projection(context)
        return x + external.unsqueeze(-1).unsqueeze(-1)

    def forward(self, att_nbhd_inputs, att_flow_inputs, att_lstm_inputs, 
                nbhd_inputs, flow_inputs, lstm_inputs, context_inputs=None,
                att_context_inputs=None):

        batch_size = lstm_inputs.shape[0]
        
        nbhd_vecs = []
        for ts in range(self.lstm_seq_len):
            context = self._context_at(context_inputs, ts)
            nbhd = F.relu(self.nbhd_convs_level0[ts](nbhd_inputs[ts]))
            flow = torch.sigmoid(self.flow_convs_level0[ts](flow_inputs[ts]))
            x = nbhd * flow
            x = self._add_context(x, context)
            
            x = F.relu(self.nbhd_convs_level1[ts](x))
            flow = torch.sigmoid(self.flow_convs_level1[ts](flow_inputs[ts]))
            x = x * flow
            x = self._add_context(x, context)
            
            x = F.relu(self.nbhd_convs_level2[ts](x))
            flow = torch.sigmoid(self.flow_convs_level2[ts](flow_inputs[ts]))
            x = x * flow
            x = self._add_context(x, context)
            
            x = x.reshape(batch_size, -1)
            x = F.relu(self.nbhd_dense[ts](x))
            x = self.dropout(x)
            nbhd_vecs.append(x)
        
        nbhd_vec = torch.stack(nbhd_vecs, dim=1)
        lstm_input = torch.cat([lstm_inputs, nbhd_vec], dim=-1)
        
        lstm_output, _ = self.short_term_lstm(lstm_input)
        lstm_output = lstm_output[:, -1, :]
        lstm_output = self.dropout(lstm_output)
        
        att_outputs = []
        for att in range(self.att_lstm_num):
            seq = []
            for ts in range(self.att_lstm_seq_len):
                idx = att * self.att_lstm_seq_len + ts
                context = self._att_context_at(att_context_inputs, att, ts, idx)
                
                nbhd = F.relu(self.att_nbhd_convs_level0[att][ts](att_nbhd_inputs[idx]))
                flow = torch.sigmoid(self.att_flow_convs_level0[att][ts](att_flow_inputs[idx]))
                x = nbhd * flow
                x = self._add_context(x, context)
                
                x = F.relu(self.att_nbhd_convs_level1[att][ts](x))
                flow = torch.sigmoid(self.att_flow_convs_level1[att][ts](att_flow_inputs[idx]))
                x = x * flow
                x = self._add_context(x, context)
                
                x = F.relu(self.att_nbhd_convs_level2[att][ts](x))
                flow = torch.sigmoid(self.att_flow_convs_level2[att][ts](att_flow_inputs[idx]))
                x = x * flow
                x = self._add_context(x, context)
                
                x = x.reshape(batch_size, -1)
                x = F.relu(self.att_nbhd_dense[att][ts](x))
                x = self.dropout(x)
                seq.append(x)
            
            seq = torch.stack(seq, dim=1)
            att_lstm_input = torch.cat([att_lstm_inputs[att], seq], dim=-1)
            out, _ = self.att_lstms[att](att_lstm_input)
            out = self.dropout(out)
            att_outputs.append(out)
        
        att_low = []
        for i in range(self.att_lstm_num):
            out = self.attention_layers[i]([att_outputs[i], lstm_output])
            att_low.append(out)
        
        att_low = torch.stack(att_low, dim=1)
        att_high, _ = self.attention_lstm(att_low)
        att_high = att_high[:, -1, :]
        att_high = self.dropout(att_high)
        
        final = torch.cat([att_high, lstm_output], dim=-1)
        out = torch.tanh(self.output_dense(final))
        
        return out


class models:
    def stdn(self, *args, **kwargs):
        return STDN(*args, **kwargs)
