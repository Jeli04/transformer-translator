import torch
import sentencepiece as spm
from transformer import Transformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(0)

batch_size = 64 # number of sequences per batch
block_size = 256 # max length of sequence/characters per sequence
max_iters = 5000 # number of iterations to train for
eval_iters = 200 # iterations to evaluate model performance
eval_interval = 250 # interval to evaluate model performance
learning_rate = 3e-4
n_embed = 384 # embedding dimension
n_head = 6 # number of heads
head_size = n_embed // n_head # size of each head
n_layer = 6 # number of layers
dropout = 0.2

sp = spm.SentencePieceProcessor()
sp.Load("models/sentencepiece_model.model")
vocab_size_x = len(sp)
vocab_size_y = len(sp)

model = Transformer(n_embed, n_head, dropout, block_size, vocab_size_x, vocab_size_y, n_layer).to(device)
checkpoint = torch.load("models/test_model.pth")
model.load_state_dict(checkpoint)

en = torch.tensor(sp.EncodeAsIds("Hello"), dtype=torch.long, device=device)
print(model.generate(en, 10))
