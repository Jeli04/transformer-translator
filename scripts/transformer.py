import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import math
from torch.nn import functional as F
import matplotlib.pyplot as plt

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
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         self.d_model = d_model

#         # d_model is the dimension of the embedding vector
#         position = torch.arange(0, max_len, device=device).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
#         self.pe = torch.zeros(max_len, 1, d_model, device=device)
#         self.pe[:, 0, 0::2] = torch.sin(position * div_term)
#         self.pe[:, 0, 1::2] = torch.cos(position * div_term)

#     def forward(self, x):
#         x = x * math.sqrt(self.d_model)
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)
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
    Multi Head Attention Class
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, n_embed):
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
            wei = q @ k.transpose(-2, -1) * (C/self.num_heads) **-0.5  # C is head size

            if mask!=None:
                mask = mask.unsqueeze(1)
                # print("wei: ", wei.shape)
                wei = wei.masked_fill(mask==0, -1 * 1e9) # mask out the upper triangle (B, T, T)
            # if self.counter % 250 == 0:
            #     self.visualize_attention(wei)

            wei = F.softmax(wei, dim=-1)  # normalize

            out = wei @ v

            # out = out.transpose(1, 2).contiguous().view(B,T,C)  # re-assemble all head outputs side by side after spliting into batches
            out = out.transpose(1, 2)\
            .contiguous().view(query.shape[0], -1, self.n_embed) # (B, L, d_model)            # if mask==None: print("Encoder: ")
            # else: print("Decoder: ")
            # print(out)
        self.counter+=1
        out = self.dropout(self.proj(out))
        return out


"""
    Feed Forward Class
"""
class FeedForward(nn.Module):
    def __init__(self, n_embed):
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
    def __init__(self, n_embed, n_head, block_size):
        super().__init__()
        self.n_embed = n_embed
        self.n_head = n_head

        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size)))) # a buffer is a tensor in a PyTorch module that isn't a model parameter but still used in the module

        self.sa_heads = MultiHeadAttention(n_head, n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.cross_attention = MultiHeadAttention(n_head, n_embed) 
        self.ln2 = nn.LayerNorm(n_embed)
        self.ffwd = FeedForward(n_embed)
        self.ln3 = nn.LayerNorm(n_embed)

    def forward(self, x, enc_output, enc_mask, dec_mask, c_attn_mask):
        B, T, C= x.shape

        x_1 = self.ln1(x)
        x = x + self.sa_heads(x_1, x_1, x_1, dec_mask)
        x_2 = self.ln2(x)
        x = x + self.cross_attention(x_2, enc_output, enc_output, c_attn_mask)
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
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.n_embed = n_embed
        self.n_head = n_head
        # self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size)))) # a buffer is a tensor in a PyTorch module that isn't a model parameter but still used in the module

        self.sa_heads = MultiHeadAttention(n_head, n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ffwd = FeedForward(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, enc_mask):
        B, T, C= x.shape

        x_1 = self.ln1(x)
        x = x + self.sa_heads(x_1, x_1, x_1, enc_mask)
        x = x + self.ffwd(self.ln2(x))
        return x
    

"""
    Decoder Block Class
"""
class DecoderBlock(nn.Module):
    def __init__(self, n_embed, n_head, block_size, n_layers):
        super().__init__()
        self.ln = nn.LayerNorm(n_embed)
        self.n_layers = n_layers
        self.decoder_layers = nn.ModuleList([Decoder(n_embed, n_head, block_size) for _ in range(n_layers)])

    def forward(self, x, enc_output, enc_mask, dec_mask, c_attn_mask):
        for i in range(self.n_layers):
            x = self.decoder_layers[i](x, enc_output, enc_mask, dec_mask, c_attn_mask)
        return self.ln(x)
    

"""
    Encoder Block Class
"""
class EncoderBlock(nn.Module):
    def __init__(self, n_embed, n_head, block_size, n_layers):
        super().__init__()
        self.ln = nn.LayerNorm(n_embed)
        self.n_layers = n_layers
        self.encoder_layers = nn.ModuleList([Encoder(n_embed, n_head) for _ in range(n_layers)])

    def forward(self, x, enc_mask):
        for i in range(self.n_layers):
            x = self.encoder_layers[i](x, enc_mask)
        return self.ln(x)


"""
    Transformer Class
"""
class Transformer(nn.Module):
    def __init__(self, n_embed, n_head, block_size, vocab_size_x, vocab_size_y, n_layer):
        super().__init__()
        self.n_embed = n_embed
        self.n_head = n_head
        # self.dropout = dropout
        self.block_size = block_size

        self.token_embedding_table_x = nn.Embedding(vocab_size_x, n_embed) # paramters are num_embeddings (size of dictionary), embedding_dim (dim of embeddign vec)
        self.token_embedding_table_y = nn.Embedding(vocab_size_y, n_embed) # paramters are num_embeddings (size of dictionary), embedding_dim (dim of embeddign vec)
        self.pos_enc = PositionalEncoding(n_embed)
        self.decoder_block = DecoderBlock(n_embed, n_head, block_size, n_layer)
        self.encoder_block = EncoderBlock(n_embed, n_head, block_size, n_layer)

        self.lm_head = nn.Linear(n_embed, vocab_size_y) # paramters are in_features, out_features

    def forward(self, x, targets, src_mask, target_mask, c_attn_mask):

        x_tok_emb = self.token_embedding_table_x(x)
        y_tok_emb = self.token_embedding_table_y(targets)

        x_pos_enc = self.pos_enc(x_tok_emb)
        y_pos_enc = self.pos_enc(y_tok_emb)
        
        # print("target shape: ", y_pos_enc.shape)

        enc_output = self.encoder_block(x_pos_enc, src_mask)
        dec_output = self.decoder_block(y_pos_enc, enc_output, src_mask, target_mask, c_attn_mask)

        output = nn.LogSoftmax(dim=-1)(self.lm_head(dec_output))

        # logits = self.lm_head(dec_output)

        # if targets == None:
        #     loss = None
        # else:
        #     # pytorch wants a (B, C, T) tensor for cross_entropy so we need some reshaping
        #     B, T, C = logits.shape
        #     # print("Logits: ", logits.shape)
        #     # print("Targets: ", targets.shape)
        #     # print(targets)
        #     logits = logits.view(B*T, C)
        #     targets = targets.view(B*T)
        #     # torch.set_printoptions(threshold=10_000)
        #     # print("Logits: ", logits)
        #     # print("Targets: ", targets)
        #     loss = F.cross_entropy(logits, targets)

        # print(logits.shape)
        return output


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
        global dropout
        dropout = 0 # turn off dropout

        self.eval() # put model in eval mode
        softmax = nn.LogSoftmax(dim=-1)

        src = torch.stack([src])
        src_mask = (src != 0).unsqueeze(1).to(device=device)  # (B, 1, L)
        src_tok_emb = self.token_embedding_table_x(src)
        src_pos_enc = self.pos_enc(src_tok_emb)

        enc_output = self.encoder_block(src_pos_enc, src_mask)

        # Define the file path where you want to save the tensor
        file_path = "tensor_data.txt"

        # Save the tensor to the text file
        with open(file_path, 'w') as file:
            for row in enc_output[0]:  # Iterate through the 96 tensors
                row_str = ' '.join(map(str, row.tolist()))  # Convert row tensor to space-separated string
                file.write(f"{row_str}\n")  # Write each row to the file

        # Confirm that the tensor has been saved
        print(f"Tensor saved to {file_path}")

        target = torch.zeros(seq_len).long().to(device)
        target[0] = 1   # set the first token to be the start token
        target = torch.stack([target])
        target_len = 0

        for i in range(1, seq_len):
            trg_mask = (target != 0).unsqueeze(1)  # (B, 1, L)
            c_attn_mask = src_mask.expand(-1, seq_len, -1) & trg_mask.expand(-1, -1, seq_len)  # (B, L, L)

            nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool)  # (1, L, L)
            nopeak_mask = torch.tril(nopeak_mask).to(device)  # (1, L, L) to triangular shape
            trg_mask = (trg_mask & nopeak_mask) # (B, L, L) padding false


            target_tok_emb = self.token_embedding_table_y(target)
            target_pos_enc = self.pos_enc(target_tok_emb)

            dec_output = self.decoder_block(target_pos_enc, enc_output, src_mask, trg_mask, c_attn_mask)

            output = softmax(self.lm_head(dec_output))
            output = torch.argmax(output, dim=-1) # (1, seq_len)

            # print(res)

            last_word_id = output[0][i].item()
            # if last_word_id == 0:
                # torch.set_printoptions(threshold=100_000)
                # print(output.dtype)
                # torch.set_printoptions(profile="default") # reset

            # print(last_word_id)
            # print(res[i])
            
            # print(last_word_id)
            target[0][i] = last_word_id
            target_len = i

            if last_word_id == 2:
                break

        print(target[0])
        dropout = 0.2 # turn dropout back on
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
