import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Attention layer for PyTorch
    Supports three attention methods: 'lba' (location-based), 'ga' (general), 'cba' (concat-based)
    """
    def __init__(self, method=None):
        super(Attention, self).__init__()
        if method != 'lba' and method != 'ga' and method != 'cba' and method is not None:
            raise ValueError('attention method is not supported')
        self.method = method
        self.att_size = None
        self.query_dim = None
        self.Wq = None
        self.Wh = None
        self.v = None

    def build(self, input_shape):
        """
        Build the attention layer with appropriate weights
        input_shape can be a tuple (batch, time, features) or list of two tuples
        """
        if isinstance(input_shape, list):
            self.att_size = input_shape[0][-1]
            self.query_dim = input_shape[1][-1]
            if self.method == 'ga' or self.method == 'cba':
                self.Wq = nn.Parameter(torch.randn(self.query_dim, self.att_size) * 0.01)
                nn.init.xavier_normal_(self.Wq.data)
        else:
            self.att_size = input_shape[-1]

        if self.method == 'cba':
            self.Wh = nn.Parameter(torch.randn(self.att_size, self.att_size) * 0.01)
            nn.init.xavier_normal_(self.Wh.data)
        
        if self.method == 'lba' or self.method == 'cba':
            self.v = nn.Parameter(torch.zeros(self.att_size, 1))

    def forward(self, inputs, mask=None):
        """
        Forward pass for attention
        :param inputs: list of [memory, query] or just memory tensor (batch, time, features)
        :param mask: optional mask tensor
        :return: weighted sum along time dimension
        """
        if isinstance(inputs, list) and len(inputs) == 2:
            memory, query = inputs
            if self.method is None:
                return memory[:, -1, :]
            elif self.method == 'cba':
                # memory: (batch, time, att_size), query: (batch, query_dim)
                hidden = torch.matmul(memory, self.Wh) + torch.matmul(query, self.Wq).unsqueeze(1)
                hidden = torch.tanh(hidden)
                s = torch.matmul(hidden, self.v).squeeze(-1)  # (batch, time)
            elif self.method == 'ga':
                s = torch.sum(torch.matmul(query, self.Wq).unsqueeze(1) * memory, dim=-1)
            else:  # 'lba'
                s = torch.matmul(memory, self.v).squeeze(-1)
            
            if mask is not None:
                mask = mask[0] if isinstance(mask, list) else mask
        else:
            if isinstance(inputs, list):
                if len(inputs) != 1:
                    raise ValueError('inputs length should not be larger than 2')
                memory = inputs[0]
            else:
                memory = inputs
            
            if self.method is None:
                return memory[:, -1, :]
            elif self.method == 'cba':
                hidden = torch.matmul(memory, self.Wh)
                hidden = torch.tanh(hidden)
                s = torch.matmul(hidden, self.v).squeeze(-1)
            elif self.method == 'ga':
                raise ValueError('general attention needs the second input')
            else:  # 'lba'
                s = torch.matmul(memory, self.v).squeeze(-1)
            
            mask = None

        s = F.softmax(s, dim=-1)
        if mask is not None:
            s = s * mask.float()
            sum_by_time = s.sum(dim=-1, keepdim=True)
            s = s / (sum_by_time + 1e-7)
        
        return torch.sum(memory * s.unsqueeze(-1), dim=1)


class SimpleAttention(nn.Module):
    """
    Simple Attention layer for PyTorch that concatenates attention with last output
    """
    def __init__(self, method=None):
        super(SimpleAttention, self).__init__()
        if method != 'lba' and method != 'ga' and method != 'cba' and method is not None:
            raise ValueError('attention method is not supported')
        self.method = method
        self.att_size = None
        self.query_dim = None
        self.Wq = None
        self.Wh = None
        self.v = None

    def build(self, input_shape):
        """
        Build the simple attention layer
        """
        if isinstance(input_shape, list):
            self.att_size = input_shape[0][-1]
            self.query_dim = input_shape[1][-1] + self.att_size
        else:
            self.att_size = input_shape[-1]
            self.query_dim = self.att_size

        if self.method == 'cba' or self.method == 'ga':
            self.Wq = nn.Parameter(torch.randn(self.query_dim, self.att_size) * 0.01)
            nn.init.xavier_normal_(self.Wq.data)
        
        if self.method == 'cba':
            self.Wh = nn.Parameter(torch.randn(self.att_size, self.att_size) * 0.01)
            nn.init.xavier_normal_(self.Wh.data)

        if self.method == 'lba' or self.method == 'cba':
            self.v = nn.Parameter(torch.zeros(self.att_size, 1))

    def forward(self, inputs, mask=None):
        """
        Forward pass
        :param inputs: list of [memory, query] or just memory tensor
        :param mask: optional mask
        :return: concatenated attention output and last output
        """
        query = None
        if isinstance(inputs, list):
            memory = inputs[0]
            if len(inputs) > 1:
                query = inputs[1]
            elif len(inputs) > 2:
                raise ValueError('inputs length should not be larger than 2')
            if isinstance(mask, list):
                mask = mask[0]
        else:
            memory = inputs

        input_shape = memory.shape
        reshape_flag = False
        if len(input_shape) > 3:
            reshape_flag = True
            input_length = input_shape[1]
            batch_size = input_shape[0]
            memory = memory.reshape(-1, input_shape[2], input_shape[3])
            if mask is not None:
                mask = mask.reshape(-1, input_shape[2])

        last = memory[:, -1, :]
        memory_seq = memory[:, :-1, :]
        
        if query is None:
            query = last
        else:
            query = torch.cat([query, last], dim=-1)

        if self.method is None:
            if reshape_flag:
                return last.reshape(batch_size, input_length, -1)
            else:
                return last
        elif self.method == 'cba':
            hidden = torch.matmul(memory_seq, self.Wh) + torch.matmul(query, self.Wq).unsqueeze(1)
            hidden = torch.tanh(hidden)
            s = torch.matmul(hidden, self.v).squeeze(-1)
        elif self.method == 'ga':
            s = torch.sum(torch.matmul(query, self.Wq).unsqueeze(1) * memory_seq, dim=-1)
        else:  # 'lba'
            s = torch.matmul(memory_seq, self.v).squeeze(-1)

        s = F.softmax(s, dim=-1)
        if mask is not None:
            mask = mask[:, :-1]
            s = s * mask.float()
            sum_by_time = s.sum(dim=-1, keepdim=True)
            s = s / (sum_by_time + 1e-7)
        
        result = torch.cat([torch.sum(memory_seq * s.unsqueeze(-1), dim=1), last], dim=-1)
        
        if reshape_flag:
            return result.reshape(batch_size, input_length, -1)
        else:
            return result
