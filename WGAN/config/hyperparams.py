import torch
import os

experiment = 'cn10_01_crit5_SA'
os.makedirs(f'{experiment}/logs', exist_ok=True)
os.makedirs(f'{experiment}/checkpoints', exist_ok=True)

# model parameters
critic_iterations = 5   # [1, 10] # 10 for complex datasets
lr = 0.00025            # casa dos 1 ou 2 10-4
gp_lambda = 10          # 10 no paper original [5,15]
adv_lambda = 0.01       # [0.01, 0.001]
content_lambda = 10      # [1, 10] # antes tava 5 
percep_lambda = 0.1     # [0.1, 1] 

# train parameters
epochs = 1000
device = torch.device("cuda:0")
batch_size = 32
save_every = 10


# TODO IMPLEMENTAR!! E TESTAR DIFERENTES KERNEL SIZES. PFS Tambem (?)
# frequency separation parameters
# freq_sep = False
# filter_size = 5
# padding = filter_size // 2
# low = nn.AvgPool2d(filter_size, stride=1, padding=0)
# rf = nn.ReplicationPad2d(padding)

