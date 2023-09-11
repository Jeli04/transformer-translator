import torch
import sentencepiece as spm
from transformer import Transformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(0)

batch_size = 64 # number of sequences per batch
block_size = 96 # max length of sequence/characters per sequence
max_iters = 2000 # number of iterations to train for
eval_iters = 200 # iterations to evaluate model performance
eval_interval = 250 # interval to evaluate model performance
learning_rate = 1e-6
n_embed = 384 # embedding dimension
n_head = 8 # number of heads
head_size = n_embed // n_head # size of each head
n_layer = 6 # number of layers
dropout = 0.2
translate_interval = 100

en_sp = spm.SentencePieceProcessor()
en_sp.Load("models/sentencepiece_model_10k_english2.model")

es_sp = spm.SentencePieceProcessor()
es_sp.Load("models/sentencepiece_model_10k_spanish.model")

vocab_size_x = len(en_sp)
vocab_size_y = len(es_sp)

model = Transformer(n_embed, n_head, block_size, vocab_size_x, vocab_size_y, n_layer).to(device)
checkpoint = torch.load("models/model_two_tok.pth", strict=False)
model.load_state_dict(checkpoint)

en = torch.tensor(en_sp.EncodeAsIds("How are you?"), dtype=torch.long, device=device)
es = torch.tensor(es_sp.EncodeAsIds("Cómo estás"), dtype=torch.long, device=device)

print(en)
print(es)

bos = es_sp.bos_id()  
eos = es_sp.eos_id()
en = torch.cat([torch.tensor([bos], device=device), en, torch.tensor([eos], device=device)]) 

len1 = en.size(0)
len2 = 96   # block size

# Pad tensors if needed
if len1 < len2:
    padding = torch.zeros(len2 - len1, device=device , dtype=torch.long)
    en = torch.cat([en, padding])

print(en)

print(en_sp.DecodeIds(model.generate(en, len2).tolist()))

