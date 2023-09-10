import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import math
from torch.nn import functional as F
import matplotlib.pyplot as plt


batch_size = 64 # number of sequences per batch
block_size = 256 # max length of sequence/characters per sequence
max_iters = 5000 # number of iterations to train for
eval_interval = 200 # interval to evaluate model performance
learning_rate = 3e-4
n_embed = 384 # embedding dimension
n_head = 6 # number of heads
head_size = n_embed // n_head # size of each head
n_layer = 6 # number of layers
dropout = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(0)

"""
    Positonal Encoding Class
    Input:
        The entire sequence of words

    Output:
        The entire sequence of words with positional encoding
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # d_model is the dimension of the embedding vector
        position = torch.arange(0, max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, 1, d_model, device=device)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


"""
    Multi Head Attention Class
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, n_embed, dropout):
        super().__init__()
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(p=dropout)

        self.key = nn.Linear(n_embed, n_embed, bias=False)
        self.query = nn.Linear(n_embed, n_embed, bias=False)
        self.value = nn.Linear(n_embed, n_embed, bias=False)

        self.num_heads = num_heads
        self.n_embed = n_embed

        self.counter = 0

    def visualize_attention(self, wei):
        # Plot attention scores
        layer_num = 1  # Specify the layer number you want to visualize
        head_num = 0   # Specify the attention head you want to visualize

        # Get the attention scores for the specified layer and head
        attention_map = wei[0][0].cpu().detach().numpy()
        # print(attention_map.shape)

        # Plot the attention heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(attention_map, cmap='viridis', aspect='auto')
        plt.title(f"Layer {layer_num}, Head {head_num} Attention")
        plt.xlabel("Input Tokens")
        plt.ylabel("Output Tokens")
        plt.colorbar()
        plt.show()


    def forward(self, query, key, value, mask=None):
        B, T, C = query.shape  # B is batch size, T is sequence length, C is n_embed
        # d_k is n_embed

        # Perform linear transformation
        q, k, v = self.query(query), self.key(key), self.value(value)
        # k = k.view(B, T, self.n_embed, C // self.n_embed).transpose(1, 2)  # (B, num_heads, T, head_size)
        # q = q.view(B, T, self.n_embed, C // self.n_embed).transpose(1, 2)  # (B, num_heads, T, head_size)
        # v = v.view(B, T, self.n_embed, C // self.n_embed).transpose(1, 2)  # (B, num_heads, T, head_size)
        k = k.view(B, -1, self.num_heads, self.n_embed // self.num_heads).transpose(1, 2)  # (B, num_heads, T, head_size)
        q = q.view(B, -1, self.num_heads, self.n_embed // self.num_heads).transpose(1, 2)  # (B, num_heads, T, head_size)
        v = v.view(B, -1, self.num_heads, self.n_embed // self.num_heads).transpose(1, 2)  # (B, num_heads, T, head_size)
        with torch.cuda.amp.autocast_mode.autocast(enabled=False):

            wei = q @ k.transpose(-2, -1) * C **-0.5  # C is head size

            if mask!=None:
                mask = mask.unsqueeze(1)
                wei = wei.masked_fill(mask==0, -1 * 1e9) # mask out the upper triangle (B, T, T)
                # print(wei)
            # if self.counter % 250 == 0:
            #     self.visualize_attention(wei)

            wei = F.softmax(wei, dim=-1)  # normalize

            out = wei @ v

            out = out.transpose(1, 2).contiguous().view(B,T,C)  # re-assemble all head outputs side by side after spliting into batches

            # if mask==None: print("Encoder: ")
            # else: print("Decoder: ")
            # print(out)
        self.counter+=1
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
        Cross Attention output
"""
class Decoder(nn.Module):
    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()
        self.n_embed = n_embed
        self.n_head = n_head

        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size)))) # a buffer is a tensor in a PyTorch module that isn't a model parameter but still used in the module

        self.sa_heads = MultiHeadAttention(n_head, n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.cross_attention = MultiHeadAttention(n_head, n_embed, dropout) 
        self.ln2 = nn.LayerNorm(n_embed)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln3 = nn.LayerNorm(n_embed)

    def forward(self, x, enc_output, enc_mask, dec_mask):
        B, T, C= x.shape

        x_1 = self.ln1(x)
        x = x + self.sa_heads(x_1, x_1, x_1, dec_mask)
        x_2 = self.ln2(x)
        x = x + self.cross_attention(x_2, enc_output, enc_output, enc_mask)
        x = x + self.ffwd(self.ln3(x))
        
        return x

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
        # self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size)))) # a buffer is a tensor in a PyTorch module that isn't a model parameter but still used in the module

        self.sa_heads = MultiHeadAttention(n_head, n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, enc_mask):
        B, T, C= x.shape

        x_1 = self.ln1(x)
        x = x + self.sa_heads(x_1, x_1, x_1, enc_mask)
        x = x + self.ffwd(self.ln2(x))
        return x

"""
    Transformer Class
"""
class Transformer(nn.Module):
    def __init__(self, n_embed, n_head, dropout, block_size, vocab_size_x, vocab_size_y, n_layer):
        super().__init__()
        self.n_embed = n_embed
        self.n_head = n_head
        self.dropout = dropout
        self.block_size = block_size

        self.token_embedding_table_x = nn.Embedding(vocab_size_x, n_embed) # paramters are num_embeddings (size of dictionary), embedding_dim (dim of embeddign vec)
        self.token_embedding_table_y = nn.Embedding(vocab_size_y, n_embed) # paramters are num_embeddings (size of dictionary), embedding_dim (dim of embeddign vec)
        self.pos_enc = PositionalEncoding(n_embed, dropout)
        self.encoder_layers = nn.ModuleList([Encoder(n_embed, n_head, dropout) for _ in range(n_layer)])
        self.decoder_layers = nn.ModuleList([Decoder(n_embed, n_head, block_size, dropout) for _ in range(n_layer)])

        # self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head, block_size=block_size, dropout=dropout) for _ in range(n_layer)]) # shortened way for multiple blocks in a sequential model
        self.ln_f = nn.LayerNorm(n_embed)    # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size_y) # paramters are in_features, out_features

    def forward(self, x, targets, src_mask, target_mask):

        x_tok_emb = self.token_embedding_table_x(x)
        y_tok_emb = self.token_embedding_table_y(targets)

        x_pos_enc = self.pos_enc(x_tok_emb)
        y_pos_enc = self.pos_enc(y_tok_emb)
        
        # print("target shape: ", y_pos_enc.shape)

        enc_output = x_pos_enc
        for encoder in self.encoder_layers:
          enc_output = encoder(enc_output, src_mask)

        dec_output = y_pos_enc
        for decoder in self.decoder_layers:
          dec_output = decoder(dec_output, enc_output, src_mask, target_mask) 
        
        # print("before final linear forward shape: ", dec_output.shape)


        res = self.ln_f(dec_output)
        logits = self.lm_head(res)

        if targets == None:
            loss = None
        else:
            # pytorch wants a (B, C, T) tensor for cross_entropy so we need some reshaping
            B, T, C = logits.shape
            # print("Logits: ", logits.shape)
            # print("Targets: ", targets.shape)
            # print(targets)
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # torch.set_printoptions(threshold=10_000)
            # print("Logits: ", logits)
            # print("Targets: ", targets)
            loss = F.cross_entropy(logits, targets)

        # print(logits.shape)
        return logits, loss


    """
        Generate function
        tokenizes the input sentence first 
        takes a source and pos encodes and embed it 
        send it to the encoder 
        take a empty target of length of input sentence and pos encode and embed it
        send it to the decoder 
        cross attnetion between encoder and decoder outputs 
        softmax the result 

    """
    # the job of generate is to extend idx to be B by T+1, B by T+2 ....
    def generate(self, src, seq_len):
        # self.dropout = 0 # turn off dropout

        # self.eval() # put model in eval mode

        # src = torch.stack([src])
        # src = self.token_embedding_table_x(src)
        # src = self.pos_enc(src)
        # enc_output = src
        # for encoder in self.encoder_layers:
        #   enc_output = encoder(enc_output)

        # # print(enc_output.shape)

        # target = torch.zeros(seq_len).long().to(device)
        # target[0] = 1   # set the first token to be the start token
        # target = torch.stack([target])
        # target_len = 0

        # for i in range(1, seq_len):
        #     # print(target)
        #     target_embed = self.token_embedding_table_y(target)
        #     target_enc = self.pos_enc(target_embed)

        #     # print("target shape: ", target_enc.shape)

        #     dec_output = target_enc
        #     for decoder in self.decoder_layers:
        #         dec_output = decoder(dec_output, enc_output)
        
        #     dec_output = self.ln_f(dec_output)
        #     dec_output = self.lm_head(dec_output)
        #     res = F.softmax(dec_output, dim=-1)

        #     # res = F.softmax(self.lm_head(self.ln_f(dec_output)), dim=-1)

        #     # added this to make the result match logits (batch_size * seq_len, vocab_size)
        #     B, T, C = res.shape
        #     res = res.view(B*T, C)

        #     # print("dec shape:", res.shape)
        #     res = torch.argmax(res, dim=-1) # (1, seq_len)

        #     # print(res)

        #     last_word_id = res[i-1]
        #     # print(last_word_id)
            
        #     if last_word_id == 2:
        #         break
            
        #     # print(last_word_id)
        #     target[0][i] = last_word_id
        #     target_len = i

        # print(target[0])
        # self.dropout = dropout # turn dropout back on
        # self.train()
        # return target

        self.dropout = 0 # turn off dropout

        self.eval() # put model in eval mode

        src = torch.stack([src])
        src_mask = (src != 0).unsqueeze(1).to(device=device)  # (B, 1, L)


        target = torch.zeros(seq_len).long().to(device)
        target[0] = 1   # set the first token to be the start token
        target = torch.stack([target])
        target_len = 0


        for i in range(1, seq_len):
            trg_mask = (target != 0).unsqueeze(1)  # (B, 1, L)
            nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool)  # (1, L, L)
            nopeak_mask = torch.tril(nopeak_mask).to(device)  # (1, L, L) to triangular shape
            trg_mask = (trg_mask & nopeak_mask) # (B, L, L) padding false

            logits, loss = self(src, target, src_mask, trg_mask)
            # print("loss: ", loss)

            # print("dec shape:", res.shape)
            res = torch.argmax(logits, dim=-1) # (1, seq_len)

            # print(res)

            last_word_id = res[i-1]
            # print(last_word_id)
            # print(res[i])
            
            # print(last_word_id)
            target[0][i] = last_word_id
            target_len = i

            if last_word_id == 2:
                break

        print(target[0])
        self.dropout = dropout # turn dropout back on
        self.train()
        return target

# import sentencepiece as spm

# # Load SentencePiece tokenizer
# sp = spm.SentencePieceProcessor()
# sp.Load("models/sentencepiece_model.model")

# embedding = nn.Embedding(8000, 384)
# print(embedding(torch.tensor(sp.Encode("hi"), dtype=torch.long)))

# text_x = "Hello"
# vocab_size_x = len(sp)
# input_x = torch.tensor(sp.encode(text_x))

# text_y = "Hola"
# vocab_size_y = len(sp)
# input_y = torch.tensor(sp.encode(text_y))

# model = Transformer(n_embed, n_head, dropout, block_size, vocab_size_x, vocab_size_y, n_layer)
# print(model(input_x, input_y))


# text_x = "Hello" #there how are you doing"
# chars = sorted(list(set(text_x))) # sets gets one of each value
# stoi = {ch:i for i, ch in enumerate(chars)} # map for string to int
# encode = lambda s : [stoi[c] for c in s]  # encodes the string into ints
# data_x = torch.tensor(encode(text_x), dtype=torch.long)

# chars = sorted(list(set(text_x))) # sets gets one of each value
# vocab_size_x = len(chars)

# text_y = "Hola " # com ó estás"
# chars = sorted(list(set(text_y))) # sets gets one of each value
# stoi = {ch:i for i, ch in enumerate(chars)} # map for string to int
# encode = lambda s : [stoi[c] for c in s]  # encodes the string into ints
# data_y = torch.tensor(encode(text_y), dtype=torch.long)

# chars = sorted(list(set(text_y))) # sets gets one of each value
# vocab_size_y = len(chars)

# # vocab size is different?

# vocab_size_x = 64
# vocab_size_y = 64

# model = Transformer(n_embed, n_head, dropout, block_size, vocab_size_x, vocab_size_y, n_layer)





# # embedding_x = torch.tensor(sp.EncodeAsIds(text_x), dtype=torch.long)
# # embedding_y = torch.tensor(sp.EncodeAsIds(text_y), dtype=torch.long)

# # data_x = torch.cat([embedding_x, torch.zeros(n_embed - len(embedding_x))], dim=-1)
# # data_y = torch.cat([embedding_y, torch.zeros(n_embed - len(embedding_y))], dim=-1)

# # print(model(data_x, data_y))
