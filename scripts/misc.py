from constants import *
from torch.nn import functional as F
import torch.nn as nn
import torch
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(0)


"""
    Multi Head Attention Class
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, n_embed):
        super().__init__()
        self.proj = nn.Linear(n_embed, n_embed, bias=False)
        self.dropout = nn.Dropout(p=dropout)

        self.key = nn.Linear(n_embed, n_embed, bias=False)
        self.query = nn.Linear(n_embed, n_embed, bias=False)
        self.value = nn.Linear(n_embed, n_embed, bias=False)

        self.num_heads = num_heads
        self.n_embed = n_embed

        self.counter = 0

    def self_attention(self, q, k, v, mask=None):
        atten_wei = q @ k.transpose(-2, -1)
        atten_wei = atten_wei / math.sqrt(d_k)  

        if mask!=None:
            mask = mask.unsqueeze(1)
            atten_wei = atten_wei.masked_fill(mask==0, -1 * 1e9) # mask out the upper triangle (B, T, T)

        atten_wei = F.softmax(atten_wei, dim=-1)  # normalize
        atten_values = self.dropout(atten_wei)

        atten_values = atten_values @ v

        return atten_values, atten_wei


    def forward(self, query, key, value, mask=None):
        B, T, C = query.shape  # B is batch size, T is sequence length, C is n_embed
        # d_k is n_embed

        # Perform linear transformation
        q, k, v = self.query(query), self.key(key), self.value(value)
        k = k.view(B, -1, self.num_heads, self.n_embed // self.num_heads).transpose(1, 2)  # (B, num_heads, T, head_size)
        q = q.view(B, -1, self.num_heads, self.n_embed // self.num_heads).transpose(1, 2)  # (B, num_heads, T, head_size)
        v = v.view(B, -1, self.num_heads, self.n_embed // self.num_heads).transpose(1, 2)  # (B, num_heads, T, head_size)
        
        # perform self attention
        atten_values, atten_wei = self.self_attention(q, k, v, mask)

        # out = out.transpose(1, 2).contiguous().view(B,T,C)  # re-assemble all head outputs side by side after spliting into batches
        atten_values = atten_values.transpose(1, 2)\
            .contiguous().view(query.shape[0], -1, self.n_embed) # (B, L, d_model)            

        atten_values = self.dropout(self.proj(atten_values))
        return atten_values, atten_wei


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len=96):
        super().__init__()
        self.d_model = d_model

        # Make initial positional encoding matrix with 0
        pe_matrix= torch.zeros(seq_len, d_model) # (L, d_model)

        # Calculating position encoding values
        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    pe_matrix[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                elif i % 2 == 1:
                    pe_matrix[pos, i] = math.cos(pos / (10000 ** (2 * i / d_model)))

        pe_matrix = pe_matrix.unsqueeze(0) # (1, L, d_model)
        self.positional_encoding = pe_matrix.to(device=device).requires_grad_(False)

    def forward(self, x):
        x = x * math.sqrt(self.d_model) # (B, L, d_model)
        x = x + self.positional_encoding # (B, L, d_model)

        return x
    
    
"""
    Feed Forward Class
"""
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),    # * 4 based on the transformer paper
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_embed * 4, n_embed),  # projection layer back to the residual path
        )

    def forward(self, x):
       return self.net(x)
