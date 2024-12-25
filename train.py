import torch
from WGAN.processing.dataloader import SRDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:1")

train_ds = SRDataset('data/train')
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)

val_ds = SRDataset('data/val')
val_dl = DataLoader(val_ds, batch_size=16, shuffle=True, num_workers=4)

from WGAN.networks.critic import Critic
from WGAN.networks.generator import Generator

critic = Critic().to(device)
generator = Generator().to(device)

# print number of parameters
print(f'Number of parameters in critic:    {sum(p.numel() for p in critic.parameters())}')
print(f'Number of parameters in generator: {sum(p.numel() for p in generator.parameters())}')

import WGAN.config.hyperparams as hp
from WGAN.wasserstein_gan import WassersteinGAN

G_optimizer = torch.optim.Adam(generator.parameters(), hp.lr, betas=(0.9, 0.99))
C_optimizer = torch.optim.Adam(critic.parameters(), hp.lr, betas=(0.9, 0.99))

trainer = WassersteinGAN(
        generator,
        critic,
        G_optimizer,
        C_optimizer
    )

trainer.train(train_dl, val_dl)