from transformer import Transformer
import preprocessing
import sentencepiece as spm
import torch
import random
import time
import gc

batch_size = 32 # number of sequences per batch
block_size = 96 # max length of sequence/characters per sequence
max_iters = 2000 # number of iterations to train for
eval_iters = 200 # iterations to evaluate model performance
eval_interval = 250 # interval to evaluate model performance
# max_iters = 20 # number of iterations to train for
# eval_iters = 5 # iterations to evaluate model performance
# eval_interval = 4 # interval to evaluate model performance
learning_rate = 1e-6
n_embed = 384 # embedding dimension
n_head = 8 # number of heads
head_size = n_embed // n_head # size of each head
n_layer = 6 # number of layers
dropout = 0.2
translate_interval = 100

start_time = time.time()
gc.collect()
torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(0)

# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device:{torch.cuda.current_device()}")
       
print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")


"""
    Split and Tokenize Data
"""
en_sp = spm.SentencePieceProcessor()
en_sp.Load("models/sentencepiece_model_10k_english2.model")

es_sp = spm.SentencePieceProcessor()
es_sp.Load("models/sentencepiece_model_10k_spanish.model")

data = preprocessing.tokenize(preprocessing.split_data(), en_sp, es_sp, block_size)

# Split proportions  
train_percent = 0.8 
val_percent = 0.2

# Calculate split sizes
train_size = int(len(data) * train_percent) 
val_size = len(data) - train_size

# Randomly shuffle data  
random.shuffle(data)

# Split data
train_data = data[:train_size]
val_data = data[train_size:]


# enc = es_sp.EncodeAsIds("si quieres sonar como un hablante nativo, debes estar dispuesto a practicar diciendo la misma frase una y otra vez de la misma manera en que un músico de banjo practica el mismo fraseo una y otra vez hasta que lo puedan tocar correctamente y en el tiempo esperado.")
# enc.insert(0, 1)
# enc.append(2)

# # Pad the list to size 96
# desired_size = 96
# if len(enc) < desired_size:
#     padding = [0] * (desired_size - len(enc))
#     enc = enc + padding

# print(enc)
# # print(train_data[0][1].tolist())

# for pair in train_data:
#     if pair[1].tolist() == enc:
#         print(pair[0])
#         print(en_sp.DecodeIds(pair[0].tolist()))

# for pair in val_data:
#     if pair[1].tolist() == enc:
#         print(pair[0])
#         print(en_sp.DecodeIds(pair[0].tolist()))

"""
    get_batch
    using the array of pairs randomly select batch size of pairs and return them
"""
def get_batch(split):
    data = train_data if split == "train" else val_data

    ix = torch.randint(len(data) - block_size, (batch_size,)) # generates n numbrs based on btach_size that represent an index from data, randomally choosing four places to parallel train

    # enc = es_sp.EncodeAsIds("no vuelvas nunca.")
    # enc.insert(0, 1)
    # enc.append(2)

    # # Pad the list to size 96
    # desired_size = 96
    # if len(enc) < desired_size:
    #     padding = [0] * (desired_size - len(enc))
    #     enc = enc + padding

    # print(enc)

    # for pair in data:
    #     if pair[1].tolist() == enc:
    #         print(pair[0])
    #         print(en_sp.DecodeIds(pair[0].tolist()))

    max_len = 0

    x_tensors = []
    y_tensors = []

    # print(ix)
    for i in ix:
      # print("Row: ", data[i])
      t = data[i][0]
      # print("X:", en_sp.DecodeIds(t.tolist()))
      x_tensors.append(t)    
      t = data[i][1]
      # print("Y:", es_sp.DecodeIds(t.tolist()))
      y_tensors.append(t)

    # for i in ix:
    #   print("Row: ", data[i])
    #   t = data[i:i+block_size][0][0]
    #   print("X:", en_sp.DecodeIds(t.tolist()))
    #   x_tensors.append(t)    
    #   t = data[i:i+block_size][0][1]
    #   print("Y:", es_sp.DecodeIds(t.tolist()))
    #   y_tensors.append(t) 

    x = torch.stack(x_tensors)      
    y = torch.stack(y_tensors)  

    x, y = x.to(device), y.to(device) # moves to GPU if available

    return x, y


"""
    estimate_loss 
    code from Andrej Karpathy's minGPT
"""
@torch.no_grad()  # tells pytorch that everything inside this function wont have back propogation
def estimate_loss():
  out = {}
  model.eval()
  for split in ["train", "val"]:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
       X, Y = get_batch(split)
       src_mask, target_mask = create_mask(X, Y)
       logits, loss = model(X, Y, src_mask, target_mask)
       losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

"""
  Create mask 
  
  creates the src and target masks
  mask out any padding tokens 
  
  Input
    the x and y batch

  Output 
    the src and target masks (B,L,L)  L = seq_len or block_size
"""
def create_mask(src, target, seq_len=block_size):
  e_mask = (src != 0).unsqueeze(1)  # (B, 1, L)
  d_mask = (target != 0).unsqueeze(1)  # (B, 1, L)

  nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool)  # (1, L, L)
  nopeak_mask = torch.tril(nopeak_mask).to(device)  # (1, L, L) to triangular shape
  d_mask = (d_mask & nopeak_mask)  # (B, L, L) padding false

  return e_mask.to(device), d_mask.to(device)


"""
    train
"""
def train_model(m):
  # print the number of parameters in the model
  print(sum(p.numel() for p in m.parameters())/1e6, "M paramters")

  # create a PyTorch optimizer
  optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

  for iter in range(max_iters):
    # once awhile evaluate the loss on train and val sets
    if iter % eval_interval == 0:
      losses = estimate_loss()  # estimate loss averages the losses of multiple batches 
      end_time = time.time()
      training_time_seconds = time.time() - start_time
      training_time_minutes = training_time_seconds // 60
      training_time_seconds %= 60
      print(f"Step: {iter} | Train loss: {losses['train']:.4f} | Val loss: {losses['val']:.4f} | Training time: {int(training_time_minutes)} minutes {int(training_time_seconds)} seconds")

      current_memory = torch.cuda.memory_allocated()
      peak_memory = torch.cuda.max_memory_allocated()
      print(f"Current memory usage: {current_memory / (1024 ** 2):.2f} MB")
      print(f"Peak memory usage: {peak_memory / (1024 ** 2):.2f} MB")

    if iter % translate_interval == 0:
      test_src = torch.tensor(en_sp.EncodeAsIds("Hello what is your name?"), dtype=torch.long, device=device)
      test_src = torch.cat([torch.tensor([1], dtype=torch.long, device=device), test_src, torch.tensor([2], dtype=torch.long, device=device)]) 

      # Pad tensors if needed
      if test_src.size(0) < block_size:
        padding = torch.zeros(block_size - test_src.size(0), dtype=torch.long, device=device)
        test_src = torch.cat([test_src, padding])

      print("Hello what is your name? : ", es_sp.DecodeIds(m.generate(test_src, block_size).tolist()))

    # clears memory
    gc.collect() 
    torch.cuda.empty_cache()

    # sample a batch of data
    xb, yb = get_batch('train')

    src_mask, trg_mask = create_mask(xb, yb)

    # evaluate the loss
    logits, loss = m.forward(xb, yb, src_mask, trg_mask)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

  end_time = time.time()
  training_time_seconds = end_time - start_time
  training_time_minutes = training_time_seconds // 60
  training_time_seconds %= 60
  print(f"Training time: {int(training_time_minutes)} minutes {int(training_time_seconds)} seconds")

  # save the model
  torch.save(m.state_dict(), "models/model_two_tok.pth")

  torch.save({
      'model_state_dict': m.state_dict(),
      "epoch": iter,
      "optimizer_state_dict": optimizer.state_dict(),
      "loss": loss
  }, "models/model_two_tok(1).pth")


vocab_size_x = len(en_sp)
vocab_size_y = len(es_sp)

model = Transformer(n_embed, n_head, dropout, block_size, vocab_size_x, vocab_size_y, n_layer).to(device)
train_model(model)

# print(get_batch('train'))
# x, y = get_batch('train')
# print(en_sp.DecodeIds(x[0].tolist()[1:-1]))
# print(es_sp.DecodeIds(y[0].tolist()[1:-1]))
# print(len(x))
# print(len(y))

# tok_data = preprocessing.tokenize(preprocessing.split_data())
# batched_data = get_batch(tok_data)
# print(batched_data[0].shape)
# print(batched_data[0])

# sp = spm.SentencePieceProcessor()
# sp.Load("models/sentencepiece_model.model")
# ids = sp.EncodeAsIds(load_data()[0][0])

# bos_id = sp.bos_id() # Get from model
# eos_id = sp.eos_id()

# ids = [bos_id] + ids + [eos_id]

# print(ids)