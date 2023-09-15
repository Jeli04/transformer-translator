batch_size = 64 # number of sequences per batch
block_size = 96 # max length of sequence/characters per sequence
max_iters = 50000 # number of iterations to train for
epochs = 15
eval_iters = 200 # iterations to evaluate model performance
eval_interval = 200 # interval to evaluate model performance
learning_rate = 1e-4
n_embed = 512 # embedding dimension
d_k = 64
d_model = 512 # dimension of model
d_ff = 2048
n_head = 8 # number of heads
head_size = n_embed // n_head # size of each head
n_layer = 6 # number of layers
dropout = 0.2
translate_interval = 100
warmup_steps = 1500
beam_length = 5

# File paths
training_file = "data/training_data.txt"
validation_file = "data/validation_data.txt"
checkpoint_folder = "models/checkpoints/"
input_file_path = 'data/spa.txt'
training_output_path = 'data/training_data.txt'
validation_output_path = 'data/validation_data.txt'
