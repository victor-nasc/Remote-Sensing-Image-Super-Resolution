import torch

# model parameters
gp_lambda = 10
critic_iterations = 5
gamma = 0.01
content_lambda = 5
lr = 0.00025

# train parameters
epochs = 1000
device = torch.device("cuda:0")
batch_size = 16
save_every = 10
log_dir = 'logs1'

# frequency separation parameters
# freq_sep = False
# filter_size = 5
# padding = filter_size // 2
# low = nn.AvgPool2d(filter_size, stride=1, padding=0)
# rf = nn.ReplicationPad2d(padding)

