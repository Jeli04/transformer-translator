from constants import *
import torch
import sentencepiece as spm
from transformer import *
from torch.nn import functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(0)

vocab_size_x = len(en_sp)
vocab_size_y = len(es_sp)

model = Transformer(n_embed, n_head, block_size, vocab_size_x, vocab_size_y, n_layer).to(device)
checkpoint = torch.load("models/checkpoints/best_ckpt.tar")
model.load_state_dict(checkpoint['model_state_dict'])

en = torch.tensor(en_sp.EncodeAsIds("Whats up"), dtype=torch.long, device=device)
es = torch.tensor(es_sp.EncodeAsIds("Cómo estás"), dtype=torch.long, device=device)

# print(en)
# print(es)

bos = es_sp.bos_id()  
eos = es_sp.eos_id()
en = torch.cat([en, torch.tensor([eos], device=device)]) 

len1 = en.size(0)
len2 = 96   # block size

# Pad tensors if needed
if len1 < len2:
    padding = torch.zeros(len2 - len1, device=device , dtype=torch.long)
    en = torch.cat([en, padding])

# print(en)
# print(es_sp.DecodeIds(model.generate(en, len2).tolist()))

x_tok_emb = model.token_embedding_table_x(en)
x_pos_enc = model.pos_enc(x_tok_emb)
e_mask = (en != 0).unsqueeze(1).to(device)
nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)  # (1, L, L)
e_mask = e_mask & nopeak_mask  # (B, L, L) padding false
e_output = model.encoder_block(x_pos_enc, e_mask)


def greedy_search(e_output, e_mask, trg_sp, model):
    last_words = torch.LongTensor([0] * seq_len).to(device) # (L)
    last_words[0] = 1 # (L)
    cur_len = 1

    for i in range(seq_len):
        d_mask = (last_words.unsqueeze(0) != 0).unsqueeze(1).to(device) # (1, 1, L)
        nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask  # (1, L, L) padding false

        trg_embedded = model.token_embedding_table_y(last_words.unsqueeze(0))
        trg_positional_encoded = model.pos_enc(trg_embedded)
        decoder_output, wei = model.decoder_block(trg_positional_encoded, e_output, e_mask, d_mask,None)


        output = F.softmax(
            model.lm_head(decoder_output), dim=-1
        ) # (1, L, trg_vocab_size)

        output = torch.argmax(output, dim=-1) # (1, L)
        last_word_id = output[0][i].item()
            
        if i < seq_len-1:
            last_words[i+1] = last_word_id
            cur_len += 1
            
        if last_word_id == 2:
            break

    if last_words[-1].item() == 0:
        decoded_output = last_words[1:cur_len].tolist()
    else:
        decoded_output = last_words[1:].tolist()
    decoded_output = trg_sp.decode_ids(decoded_output)
        
    return decoded_output


print(greedy_search(e_output, e_mask, es_sp, model))