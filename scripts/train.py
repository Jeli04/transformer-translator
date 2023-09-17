from constants import *
from transformer import Transformer
import torch.optim as optim
import preprocessing
import sentencepiece as spm
import torch
import random
import datetime
import time
import gc
from tqdm import tqdm
import sys
import os
import numpy as np
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(0)

criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
criterion.to(device)

gc.collect()
torch.cuda.empty_cache()

# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device:{torch.cuda.current_device()}")
       
print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")


"""
    Split and Tokenize Data
"""


split_data = preprocessing.split_data() # split data into training and validation files

training_dataset = preprocessing.CustomDataset(preprocessing.tokenize(preprocessing.create_pairs(training_file), en_sp, es_sp, block_size))
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

validation_dataset = preprocessing.CustomDataset(preprocessing.tokenize(preprocessing.create_pairs(validation_file), en_sp, es_sp, block_size))
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


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
  Create subsequent mask
"""
def create_subsequent_mask(target):
  """
  if target length is 5 and diagonal is 1, this function returns
      [[0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0]]
  :param target: [batch size, target length]
  :return:
  """
  batch_size, target_length = target.size()

  # torch.triu returns the upper triangular part of a matrix based on user defined diagonal
  subsequent_mask = torch.tril(torch.ones(target_length, target_length), diagonal=1).bool().to(device)
  # subsequent_mask = [target length, target length]

  # repeat subsequent_mask 'batch size' times to cover all data instances in the batch
  subsequent_mask = subsequent_mask.unsqueeze(0).repeat(batch_size, 1, 1)
  # subsequent_mask = [batch size, target length, target length]

  return subsequent_mask


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

  c_mask = (src!=0).unsqueeze(1) * (target!=0).unsqueeze(2)  

  nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)  # (1, L, L)
  e_mask = e_mask & nopeak_mask  # (B, L, L) padding false
  nopeak_mask = torch.tril(nopeak_mask).to(device)  # (1, L, L) to triangular shape
  d_mask = (d_mask & nopeak_mask)  # (B, L, L) padding false

  # print("c_mask: ", c_mask.shape)
  # print("e_mask: ", e_mask.shape)
  # print("d_mask: ", d_mask.shape)

  return e_mask.to(device), d_mask.to(device), c_mask.to(device)


"""
  Schedule learning rate
"""
class ScheduledAdam():
    def __init__(self, optimizer, hidden_dim, warm_steps):
        self.init_lr = np.power(hidden_dim, -0.5)
        self.optimizer = optimizer
        self.current_steps = 0
        self.warm_steps = warm_steps

    def step(self):
        # Update learning rate using current step information
        self.current_steps += 1
        lr = self.init_lr * self.get_scale()
        
        for p in self.optimizer.param_groups:
            p['lr'] = lr

        # print(f"Current learning rate: {lr} || Current step: {self.current_steps}")

        self.optimizer.step()
        
    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_scale(self):
        return np.min([
            np.power(self.current_steps, -0.5),
            self.current_steps * np.power(self.warm_steps, -0.5)
        ])
    

"""
  Perform validation
"""
def validation(m):
  print("Validation processing...")
  m.eval()

  valid_losses = []
  start_time = datetime.datetime.now()

  with torch.no_grad():
    for i, batch in enumerate(tqdm(training_dataloader)):
      xb, yb = batch[0], batch[1]
      xb, yb = xb.to(device), yb.to(device) # moves to GPU if available

      src_mask, trg_mask, c_mask = create_mask(xb, yb)
      # print("\n")
      # print(en_sp.DecodeIds(xb[0].tolist()))
      # print(xb[0])
      # print(es_sp.DecodeIds(yb[0].tolist()))
      # print(yb[0])
      # print("src_mask: ", src_mask[0])
      # print("trg_mask: ", trg_mask[0])

      # evaluate the loss
      output = m.forward(xb, yb, src_mask, trg_mask, c_mask)
      target_shape = yb.shape
      loss = criterion(output.view(-1, vocab_size_y), yb.view(target_shape[0] * target_shape[1]))

      valid_losses.append(loss.item())

      # clears memory
      del xb, yb, src_mask, trg_mask, output, loss
      gc.collect() 
      torch.cuda.empty_cache()


  end_time = datetime.datetime.now()
  validation_time = end_time - start_time
  seconds = validation_time.seconds
  hours = seconds // 3600
  minutes = (seconds % 3600) // 60
  seconds = seconds % 60

  mean_valid_loss = np.mean(valid_losses)

  return mean_valid_loss, f"{hours}hrs {minutes}mins {seconds}secs"
   

"""
    train
"""
def train_model(m):
  print("Training processing...")
  # print the number of parameters in the model
  print(sum(p.numel() for p in m.parameters())/1e6, "M paramters")

  # create a PyTorch optimizer
  optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

  # optimizer = ScheduledAdam(
  #           optim.Adam(m.parameters(), betas=(0.9, 0.98), eps=1e-9),
  #           hidden_dim=n_embed,
  #           warm_steps=warmup_steps
  #       )
    
  best_loss = sys.float_info.max

  for epoch in range(epochs):
    m.train()

    train_losses = []
    start_time = datetime.datetime.now()

    for i, batch in enumerate(tqdm(training_dataloader)):
      xb, yb = batch[0], batch[1]
      xb, yb = xb.to(device), yb.to(device) # moves to GPU if available

      src_mask, trg_mask, c_mask = create_mask(xb, yb)

      # evaluate the loss
      output = m.forward(xb, yb, src_mask, trg_mask, c_mask)
      optimizer.zero_grad()
      # print("output: ", output[:, 1:, :].contiguous().view(-1, output.shape[-1]).shape)
      # print("yb: ", yb[:, 1:].contiguous().view(-1).shape)
      loss = criterion(output[:, 1:, :].contiguous().view(-1, output.shape[-1]), yb[:, 1:].contiguous().view(-1))

      loss.backward()
      torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
      optimizer.step()

      train_losses.append(loss.item())

      # clears memory
      del xb, yb, src_mask, trg_mask, output, loss
      gc.collect() 
      torch.cuda.empty_cache()

    # print("Learning rate: ", optimizer.optimizer.param_groups[0]['lr'])
    print("Learning rate: ", optimizer.param_groups[0]['lr'])

    end_time = datetime.datetime.now()
    training_time = end_time - start_time
    seconds = training_time.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    mean_train_loss = np.mean(train_losses)
    print(f"#################### Epoch: {epoch} ####################")
    print(f"Train loss: {mean_train_loss} || One epoch training time: {hours}hrs {minutes}mins {seconds}secs")

    # calculate validation loss
    valid_loss, valid_time = validation(m)

    if valid_loss < best_loss:
        if not os.path.exists(checkpoint_folder):
            os.mkdir(checkpoint_folder)

        best_loss = valid_loss
        state_dict = {
            'model_state_dict': m.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'loss': best_loss
        }
        torch.save(state_dict, f"{checkpoint_folder}/best_ckpt.tar")
        print(f"***** Current best checkpoint is saved. *****")

    print(f"Best valid loss: {best_loss}")
    print(f"Valid loss: {valid_loss} || One epoch training time: {valid_time}")

    # print test translation
    test_src = torch.tensor(en_sp.EncodeAsIds("Go on."), dtype=torch.long, device=device)
    test_src = torch.cat([torch.tensor([1], dtype=torch.long, device=device), test_src, torch.tensor([2], dtype=torch.long, device=device)]) 

    # Pad tensors if needed
    if test_src.size(0) < block_size:
      padding = torch.zeros(block_size - test_src.size(0), dtype=torch.long, device=device)
      test_src = torch.cat([test_src, padding])

      print("Go on. : ", es_sp.DecodeIds(m.generate(test_src, block_size).tolist()))
      print("Go on. : ", es_sp.DecodeIds(m.beam_search(test_src, block_size, beam_length).tolist()))


  print(f"Training finished!")

      
vocab_size_x = len(en_sp)
vocab_size_y = len(es_sp)

model = Transformer(n_embed, n_head, block_size, vocab_size_x, vocab_size_y, n_layer).to(device)
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