import torch
import argparse
import WGAN.config.hyperparams as hp

from WGAN.processing.dataloader import SRDataset
from torch.utils.data import DataLoader
from WGAN.networks.critic import Critic
from WGAN.networks.generator import Generator
from WGAN.wasserstein_gan import WassersteinGAN


def main(checkpoint=0):
    train_ds = SRDataset('data/train')
    train_dl = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True, num_workers=4)

    val_ds = SRDataset('data/val')
    val_dl = DataLoader(val_ds, batch_size=hp.batch_size, shuffle=False, num_workers=4)

    critic = Critic().to(hp.device)
    generator = Generator().to(hp.device)
    print(f'Number of parameters in critic:    {sum(p.numel() for p in critic.parameters())}')
    print(f'Number of parameters in generator: {sum(p.numel() for p in generator.parameters())}')

    G_optimizer = torch.optim.Adam(generator.parameters(), hp.lr, betas=(0.9, 0.99))
    C_optimizer = torch.optim.Adam(critic.parameters(), hp.lr, betas=(0.9, 0.99))

    if checkpoint > 0:
        try:
            critic.load_state_dict(torch.load(f'{hp.experiment}/checkpoints/C_{checkpoint}.pth'))
            generator.load_state_dict(torch.load(f'{hp.experiment}/checkpoints/G_{checkpoint}.pth'))
            print(f'Loaded checkpoint {checkpoint}')
        except FileNotFoundError:
            raise FileNotFoundError(f'Checkpoint {checkpoint} not found')

    trainer = WassersteinGAN(
            generator,
            critic,
            G_optimizer,
            C_optimizer,
        )

    trainer.train(train_dl, val_dl, epoch_start=checkpoint)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="checkpoint")
    parser.add_argument('--checkpoint', type=int, default=0)
    args = parser.parse_args()
    
    main(args.checkpoint)