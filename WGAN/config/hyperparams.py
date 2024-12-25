import torch


device = torch.device("cuda:0")

# Hyper params
gp_lambda = 10
critic_iterations = 5
batch_size = 32
gamma = 0.01
content_lambda = 5
ncomp = 75
lr = 0.00025

# Run configuration parameters
epochs = 1000
print_every = 250
save_every = 10
use_cuda = True

# Frequency separation parameters
# freq_sep = False
# filter_size = 5
# padding = filter_size // 2
# low = nn.AvgPool2d(filter_size, stride=1, padding=0)
# rf = nn.ReplicationPad2d(padding)

