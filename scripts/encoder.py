from constants import * 
from misc import *
import torch.nn as nn

"""
    Encoder Class
    Input:
        Positional Encoding of data (Spanish words)
    Output:
        Multi Head Attention prediction
"""
class Encoder(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.n_embed = n_embed
        self.n_head = n_head

        self.sa_heads = MultiHeadAttention(n_head, n_embed)
        self.ln1 = nn.LayerNorm(n_embed, eps=1e-6)
        self.ffwd = FeedForward(n_embed)
        self.ln2 = nn.LayerNorm(n_embed, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, enc_mask):
        B, T, C= x.shape

        # masked multihead attention
        x_1 = self.ln1(x)
        output, wei = self.sa_heads(x_1, x_1, x_1, enc_mask)
        x = x + output

        # feed forward
        x_2 = self.ln2(x)
        x = x + self.dropout(self.ffwd(x_2))
        return x, wei
    

"""
    Encoder Block Class
"""
class EncoderBlock(nn.Module):
    def __init__(self, n_embed, n_head, block_size, n_layers):
        super().__init__()
        self.ln = nn.LayerNorm(n_embed, eps=1e-6)
        self.n_layers = n_layers
        self.encoder_layers = nn.ModuleList([Encoder(n_embed, n_head) for _ in range(n_layers)])

    def forward(self, x, enc_mask):
        for i in range(self.n_layers):
            x, wei = self.encoder_layers[i](x, enc_mask)
        return self.ln(x)
