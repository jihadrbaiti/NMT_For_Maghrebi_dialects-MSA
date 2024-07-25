import torch

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model parameter setting
batch_size = 128
max_len = 512 #the default is 256
d_model = 768
n_layers = 6
n_heads = 8
ffn_hidden = 1024
drop_prob = 0.1
max_seq_len = 30

PATH = './msa_nad/prepro/'
ext = '_data.csv'
# optimizer parameter setting
init_lr = 1e-5
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epoch = 200
clip = 1.0
weight_decay = 5e-4
inf = float('inf')