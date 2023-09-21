from constants import * 
from misc import *
import torch.nn as nn

"""
    Decoder Class
    Input:
        Positional Encoding of data (English words)
    Output:
        Cross Attention output
"""
class Decoder(nn.Module):
    def __init__(self, n_embed, n_head, block_size):
        super().__init__()
        self.n_embed = n_embed
        self.n_head = n_head

        self.sa_heads = MultiHeadAttention(n_head, n_embed)
        self.ln1 = nn.LayerNorm(n_embed, eps=1e-6)

        self.cross_attention = MultiHeadAttention(n_head, n_embed) 
        self.ln2 = nn.LayerNorm(n_embed, eps=1e-6)
         
        self.ffwd = FeedForward(n_embed)
        self.ln3 = nn.LayerNorm(n_embed, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, enc_output, enc_mask, dec_mask, c_mask):
        B, T, C= x.shape

        # masked multihead attention 
        x_1 = self.ln1(x)
        output, wei = self.sa_heads(x_1, x_1, x_1, dec_mask)
        x = x + output

        # cross attention
        x_2 = self.ln2(x)
        output, wei = self.cross_attention(x_2, enc_output, enc_output, enc_mask)
        x = x + output

        # feed forward 
        x_3 = self.ln3(x)
        x = x + self.dropout(self.ffwd(x_3))
        return x, wei
    

"""
    Decoder Block Class
"""
class DecoderBlock(nn.Module):
    def __init__(self, n_embed, n_head, block_size, n_layers):
        super().__init__()
        self.ln = nn.LayerNorm(n_embed, eps=1e-6)
        self.n_layers = n_layers
        self.decoder_layers = nn.ModuleList([Decoder(n_embed, n_head, block_size) for _ in range(n_layers)])


    def forward(self, x, enc_output, enc_mask, dec_mask, c_mask):
        for i in range(self.n_layers):
            x, wei = self.decoder_layers[i](x, enc_output, enc_mask, dec_mask, c_mask)
        
        return self.ln(x), wei
    