from transformer import Transformer
import preprocessing
import sentencepiece as spm
import torch
import random
import time

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

start_time = time.time()
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
data = preprocessing.tokenize(preprocessing.split_data())

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


"""
    get_batch
    using the array of pairs randomly select batch size of pairs and return them
"""
def get_batch(split):
    data = train_data if split == "train" else val_data

    ix = torch.randint(len(data) - block_size, (batch_size,)) # generates n numbrs based on btach_size that represent an index from data, randomally choosing four places to parallel train

    max_len = 0

    x_tensors = []
    y_tensors = []

    for i in ix:
      t = data[i:i+block_size][0][0]
      max_len = max(max_len, t.size(0))
      x_tensors.append(t)    
      t = data[i:i+block_size][0][1]
      y_tensors.append(t)

    for i in range(len(x_tensors)):
      pad_size = max_len - x_tensors[i].shape[0]

      # Create pad tensor  
      pad = torch.zeros(pad_size, dtype=torch.long)

      # Concatenate padding
      x_tensors[i] = torch.cat([x_tensors[i], pad])
      y_tensors[i] = torch.cat([y_tensors[i], pad])

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
       logits, loss = model(X, Y)
       losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out


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

      test_src = torch.tensor(sp.EncodeAsIds("Hello what is your name?"), dtype=torch.long, device=device)
      test_src = torch.cat([torch.tensor([1], dtype=torch.long, device=device), test_src, torch.tensor([2], dtype=torch.long, device=device)]) 
      print("Hello what is your name? : ", sp.DecodeIds(m.generate(test_src, 10).tolist()))

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

  end_time = time.time()
  training_time_seconds = end_time - start_time
  training_time_minutes = training_time_seconds // 60
  training_time_seconds %= 60
  print(f"Training time: {int(training_time_minutes)} minutes {int(training_time_seconds)} seconds")

  # save the model
  torch.save(m.state_dict(), "models/test_model.pth")

  torch.save({
      'model_state_dict': m.state_dict(),
      "epoch": iter,
      "optimizer_state_dict": optimizer.state_dict(),
      "loss": loss
  }, "models/test2_model.pth")


sp = spm.SentencePieceProcessor()
sp.Load("models/sentencepiece_model.model")
vocab_size_x = len(sp)
vocab_size_y = len(sp)

model = Transformer(n_embed, n_head, dropout, block_size, vocab_size_x, vocab_size_y, n_layer).to(device)
train_model(model)

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