import torch
import torch.nn as nn 
import numpy as np
import math
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device:{torch.cuda.current_device()}")
       
print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

batch_size = 64 # number of sequences per batch
block_size = 256 # max length of sequence/characters per sequence 
max_iters = 5000 # number of iterations to train for
eval_rate = 3e-4
n_embed = 384 # embedding dimension
n_head = 6 # number of heads
head_size = n_embed // n_head # size of each head
n_layer = 6 # number of layers
dropout = 0.2

"""
    Positonal Encoding Class
    Input:
        The entire sequence of words 

    Output:
        The entire sequence of words with positional encoding
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super.__init__()
        self.dropout = nn.Dropout(p=dropout)

        # d_model is the dimension of the embedding vector
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

"""
    Multi Head Attention Class
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, dropout):
        super().__init__()
        self.proj = nn.Linear(head_size, head_size)
        self.dropout = nn.Dropout(p=dropout)

        self.key = nn.Linear(head_size, head_size, bias=False)
        self.query = nn.Linear(head_size, head_size, bias=False) 
        self.value = nn.Linear(head_size, head_size, bias=False)

        self.num_heads = num_heads
        self.head_size = head_size

    def forward(self, query, key, value, mask=None):
        B, T, C = query.shape  # B is batch size, T is sequence length, C is n_embed
        # d_k is n_embed

        # Perform linear transformation
        q, k, v = self.query(query), self.key(key), self.value(value)

        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, num_heads, T, head_size)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, num_heads, T, head_size)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, num_heads, T, head_size)
        
        with torch.cuda.amp.autocast_mode.autocast(enabled=False):

            wei = q @ k.transpose(-2, -1) * C **-0.5  # C is head size 

            if mask!=None:
                mask = mask.to(device)
                wei = wei.masked_fill(mask==0, 0) # mask out the upper triangle (B, T, T)
            
            wei = F.softmax(wei, dim=-1)  # normalize

            out = wei @ v

            out = out.transpose(1, 2).contiguous().view(B,T,C)  # re-assemble all head outputs side by side after spliting into batches (97-99)

        out = self.dropout(self.proj(out))
        return out
    

"""
    Feed Forward Class
"""
class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),    # * 4 based on the transformer paper
            nn.ReLU(),
            nn.Linear(n_embed * 4, n_embed),  # projection layer back to the residual path  
            nn.Dropout(dropout),
        )

    def forward(self, x):
       return self.net(x)
    

"""
    Decoder Class
    Input: 
        Positional Encoding of data (English words)
    Output: 
        Masked Multi Head Attention prediction
"""
class Decoder(nn.Module):
    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()
        self.n_embed = n_embed
        self.n_head = n_head

        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size)))) # a buffer is a tensor in a PyTorch module that isn't a model parameter but still used in the module
        
        self.sa_heads = MultiHeadAttention(n_head, n_embed // n_head, dropout)
        self.ln1 = nn.LayerNorm(n_embed)

    def forward(self, x):
        B, T, C= x.shape

        x = x + self.sa_heads(x, x, x, self.tril[:T, :T] == 0)
        return self.ln1(x)


"""
    Encoder Class
    Input:
        Positional Encoding of data (Spanish words)
    Output:
        Multi Head Attention prediction
"""
class Encoder(nn.Module):
    def __init__(self, n_embed, n_head, dropout):
        super().__init__()
        self.n_embed = n_embed
        self.n_head = n_head
        
        self.sa_heads = MultiHeadAttention(n_head, n_embed // n_head, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        B, T, C= x.shape
        
        _x = self.ln1(x)
        x = x + self.sa_heads(_x, _x, _x, self.tril[:T, :T] == 0)
        x = x + self.ffwd(self.ln2(x))
        return x


"""
    Cross Attention Class
    Input:
        Query (English words)
        Key and Value (Spanish words)
    Output:
        Cross Attention prediction
"""
class CrossAttention(nn.Module):
    def __init__(self, n_embed, n_head, dropout):
        super().__init__()
        self.n_embed = n_embed
        self.n_head = n_head

        self.decoder = Decoder(n_embed, n_head, dropout)
        self.encoder = Encoder(n_embed, n_head, dropout)

        self.sa_heads = MultiHeadAttention(n_head, n_embed // n_head, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, y):
        k = self.encoder(x)
        v = self.decoder(x)
        q = self.decoder(y)

        res = q + self.sa_heads(q, k, v)
        res = res + self.ln1(res)
        return res + self.ffwd(self.ln2(res))


"""
    Transformer Class
"""
class Transformer(nn.Module):
    def __init__(self, n_embed, n_head, dropout, block_size, vocab_size, n_layer):
        super().__init__()
        self.n_embed = n_embed
        self.n_head = n_head
        self.dropout = dropout
        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # paramters are num_embeddings (size of dictionary), embedding_dim (dim of embeddign vec)
        self.pos_enc = PositionalEncoding(n_embed, dropout)
        self.blocks = nn.Sequential(*[CrossAttention(n_embed, n_head=n_head, dropout=dropout) for _ in range(n_layer)]) # shortened way for multiple blocks in a sequential model
        self.ln_f = nn.LayerNorm(n_embed)    # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size) # paramters are in_features, out_features

    def forward(self, x, y):
        x_tok_emb = self.token_embedding_table(x)
        y_tok_emb = self.token_embedding_table(y)

        x_pos_enc = self.pos_enc(x) # modify positional encoding so it takes raw data
        y_pos_enc = self.pos_enc(y)

        x = x_tok_emb + x_pos_enc
        y = y_tok_emb + y_pos_enc

        res = self.blocks(x, y)
        res = self.ln_f(res)
        logits = self.lm_head(res)



    