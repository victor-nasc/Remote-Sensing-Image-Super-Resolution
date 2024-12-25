import torch
import WGAN.config.hyperparams as hp

from WGAN.processing.dataloader import SRDataset
from torch.utils.data import DataLoader
from WGAN.networks.critic import Critic
from WGAN.networks.generator import Generator
from WGAN.wasserstein_gan import WassersteinGAN


def main():
    train_ds = SRDataset('data/train')
    train_dl = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=False, num_workers=4)

    val_ds = SRDataset('data/val')
    val_dl = DataLoader(val_ds, batch_size=hp.batch_size, shuffle=True, num_workers=4)

    critic = Critic().to(hp.device)
    generator = Generator().to(hp.device)
    print(f'Number of parameters in critic:    {sum(p.numel() for p in critic.parameters())}')
    print(f'Number of parameters in generator: {sum(p.numel() for p in generator.parameters())}')

    G_optimizer = torch.optim.Adam(generator.parameters(), hp.lr, betas=(0.9, 0.99))
    C_optimizer = torch.optim.Adam(critic.parameters(), hp.lr, betas=(0.9, 0.99))

    trainer = WassersteinGAN(
            generator,
            critic,
            G_optimizer,
            C_optimizer,
            log_dir=hp.log_dir,
        )

    trainer.train(train_dl, val_dl)

if __name__ == '__main__':
    main()