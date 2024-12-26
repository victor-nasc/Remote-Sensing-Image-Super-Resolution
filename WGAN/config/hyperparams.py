import torch

# model parameters
critic_iterations = 5   # [1, 10] # 10 for complex datasets
lr = 0.00025            # casa dos 1 ou 2 10-4
gp_lambda = 10          # 10 no paper original [5,15]
adv_lambda = 0.01       # [0.01, 0.001]
content_lambda = 5      # [1, 10] # antes tava 5
percep_lambda = 1     # [0.1, 1] #### TODO IMPLEMENTAR!!

# train parameters
epochs = 1000
device = torch.device("cuda:1")
batch_size = 16   
save_every = 10
log_dir = 'logs1'

# TODO IMPLEMENTAR!! E TESTAR DIFERENTES KERNEL SIZES. PFS Tambem (?)
# frequency separation parameters
# freq_sep = False
# filter_size = 5
# padding = filter_size // 2
# low = nn.AvgPool2d(filter_size, stride=1, padding=0)
# rf = nn.ReplicationPad2d(padding)

