import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, method=None, att_size=None, query_dim=None):
        super(Attention, self).__init__()
        if method not in ['lba', 'ga', 'cba', None]:
            raise ValueError('attention method is not supported')
        self.method = method

        self.att_size = None
        self.query_dim = None

        self.Wq = None
        self.Wh = None
        self.v = None

        if att_size is not None:
            if query_dim is None:
                query_dim = att_size
            self._init_parameters(att_size=att_size, query_dim=query_dim)

    def _init_parameters(self, att_size, query_dim, device=None):
        self.att_size = int(att_size)
        self.query_dim = int(query_dim)

        kwargs = {"device": device} if device is not None else {}

        if self.method in ['ga', 'cba']:
            self.Wq = nn.Parameter(torch.empty(self.query_dim, self.att_size, **kwargs))
            nn.init.xavier_normal_(self.Wq)

        if self.method == 'cba':
            self.Wh = nn.Parameter(torch.empty(self.att_size, self.att_size, **kwargs))
            nn.init.xavier_normal_(self.Wh)

        if self.method in ['lba', 'cba']:
            self.v = nn.Parameter(torch.zeros(self.att_size, 1, **kwargs))

    def build(self, memory, query):
        if self.method is None:
            return

        att_size = int(memory.shape[-1])
        query_dim = int(query.shape[-1])
        if self.att_size is not None and (self.att_size != att_size or self.query_dim != query_dim):
            raise ValueError(
                f"Attention was initialized for att_size={self.att_size}, query_dim={self.query_dim}, "
                f"but got att_size={att_size}, query_dim={query_dim}."
            )
        if self.Wq is None and self.Wh is None and self.v is None:
            self._init_parameters(att_size=att_size, query_dim=query_dim, device=memory.device)

    def forward(self, inputs, mask=None):
        if isinstance(inputs, list) and len(inputs) == 2:
            memory, query = inputs

            # 🔥 AUTO BUILD
            if self.Wh is None and self.method == 'cba':
                self.build(memory, query)
            if self.Wq is None and self.method in ['ga', 'cba']:
                self.build(memory, query)
            if self.v is None and self.method in ['lba', 'cba']:
                self.build(memory, query)

            if self.method is None:
                return memory[:, -1, :]

            elif self.method == 'cba':
                hidden = torch.matmul(memory, self.Wh) + torch.matmul(query, self.Wq).unsqueeze(1)
                hidden = torch.tanh(hidden)
                s = torch.matmul(hidden, self.v).squeeze(-1)

            elif self.method == 'ga':
                s = torch.sum(torch.matmul(query, self.Wq).unsqueeze(1) * memory, dim=-1)

            else:  # lba
                s = torch.matmul(memory, self.v).squeeze(-1)

        else:
            if isinstance(inputs, list):
                memory = inputs[0]
            else:
                memory = inputs

            if self.method is None:
                return memory[:, -1, :]

            # 🔥 AUTO BUILD
            if self.Wh is None or self.v is None:
                dummy_query = memory[:, -1, :]
                self.build(memory, dummy_query)

            if self.method == 'cba':
                hidden = torch.matmul(memory, self.Wh)
                hidden = torch.tanh(hidden)
                s = torch.matmul(hidden, self.v).squeeze(-1)

            elif self.method == 'ga':
                raise ValueError('general attention needs query')

            else:  # lba
                s = torch.matmul(memory, self.v).squeeze(-1)

        # softmax
        s = F.softmax(s, dim=-1)

        if mask is not None:
            mask = mask[0] if isinstance(mask, list) else mask
            s = s * mask.float()
            s = s / (s.sum(dim=-1, keepdim=True) + 1e-7)

        return torch.sum(memory * s.unsqueeze(-1), dim=1)
    


class SimpleAttention(nn.Module):
    def __init__(self, method=None, att_size=None, query_dim=None):
        super(SimpleAttention, self).__init__()
        if method not in ['lba', 'ga', 'cba', None]:
            raise ValueError('attention method is not supported')
        self.method = method

        self.att_size = None
        self.query_dim = None

        self.Wq = None
        self.Wh = None
        self.v = None

        if att_size is not None:
            if query_dim is None:
                query_dim = att_size
            self._init_parameters(att_size=att_size, query_dim=query_dim)

    def _init_parameters(self, att_size, query_dim, device=None):
        self.att_size = int(att_size)
        self.query_dim = int(query_dim)

        kwargs = {"device": device} if device is not None else {}

        if self.method in ['ga', 'cba']:
            self.Wq = nn.Parameter(torch.empty(self.query_dim, self.att_size, **kwargs))
            nn.init.xavier_normal_(self.Wq)

        if self.method == 'cba':
            self.Wh = nn.Parameter(torch.empty(self.att_size, self.att_size, **kwargs))
            nn.init.xavier_normal_(self.Wh)

        if self.method in ['lba', 'cba']:
            self.v = nn.Parameter(torch.zeros(self.att_size, 1, **kwargs))

    def build(self, memory, query):
        if self.method is None:
            return

        att_size = int(memory.shape[-1])
        query_dim = int(query.shape[-1])
        if self.att_size is not None and (self.att_size != att_size or self.query_dim != query_dim):
            raise ValueError(
                f"Attention was initialized for att_size={self.att_size}, query_dim={self.query_dim}, "
                f"but got att_size={att_size}, query_dim={query_dim}."
            )
        if self.Wq is None and self.Wh is None and self.v is None:
            self._init_parameters(att_size=att_size, query_dim=query_dim, device=memory.device)

    def forward(self, inputs, mask=None):
        query = None

        if isinstance(inputs, list):
            memory = inputs[0]
            if len(inputs) > 1:
                query = inputs[1]
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

        # 🔥 AUTO BUILD
        needs_build = (
            (self.method in ['ga', 'cba'] and self.Wq is None)
            or (self.method == 'cba' and self.Wh is None)
            or (self.method in ['lba', 'cba'] and self.v is None)
        )
        if needs_build:
            self.build(memory_seq, query)

        if self.method is None:
            result = last

        elif self.method == 'cba':
            hidden = torch.matmul(memory_seq, self.Wh) + torch.matmul(query, self.Wq).unsqueeze(1)
            hidden = torch.tanh(hidden)
            s = torch.matmul(hidden, self.v).squeeze(-1)

        elif self.method == 'ga':
            s = torch.sum(torch.matmul(query, self.Wq).unsqueeze(1) * memory_seq, dim=-1)

        else:  # lba
            s = torch.matmul(memory_seq, self.v).squeeze(-1)

        if self.method is not None:
            s = F.softmax(s, dim=-1)

            if mask is not None:
                mask = mask[:, :-1]
                s = s * mask.float()
                s = s / (s.sum(dim=-1, keepdim=True) + 1e-7)

            result = torch.cat([
                torch.sum(memory_seq * s.unsqueeze(-1), dim=1),
                last
            ], dim=-1)

        if reshape_flag:
            return result.reshape(batch_size, input_length, -1)
        else:
            return result
