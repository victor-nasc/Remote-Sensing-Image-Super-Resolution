import os
import torch
import WGAN.config.hyperparams as hp
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from WGAN.metrics.metrics import wass_loss, SSIM, PSNR

torch.autograd.set_detect_anomaly(True)


METRICS_TO_CALCULATE = {
    "MAE": nn.L1Loss(),
    "MSE": nn.MSELoss(),
    "Wass": wass_loss,
    "SSIM": SSIM,
    "PSNR": PSNR
}

class WassersteinGAN:
    """Implements Wasserstein GAN with gradient penalty"""

    def __init__(self, G, C, G_optimizer, C_optimizer, log_dir='logs') -> None:
        self.G = G
        self.C = C
        self.G_optimizer = G_optimizer
        self.C_optimizer = C_optimizer
        self.num_steps = 0
        self.writer = SummaryWriter(log_dir=log_dir)
        
    @staticmethod
    def _calculate_metrics(G, C, LR, HR, metrics_dict):
        fake = G(LR).detach()
        
        creal = torch.mean(C(HR)).detach()
        cfake = torch.mean(C(fake)).detach()

        for key in METRICS_TO_CALCULATE.keys():
            if key == "Wass":
                metrics_dict[key].append(METRICS_TO_CALCULATE[key](creal, cfake).detach().cpu().item())
            else:
                metrics_dict[key].append(METRICS_TO_CALCULATE[key](HR, fake).detach().cpu().item())
        
        return metrics_dict

    @staticmethod
    def _initialize_metric_dicts():
        metrics_dict = {}
        for key in METRICS_TO_CALCULATE.keys():
            metrics_dict[key] = []

        return metrics_dict

    def _tensorboard_metrics_mean(self, metrics_dict, subset_str, epoch):
        for key in METRICS_TO_CALCULATE.keys():
            metric = torch.mean(torch.FloatTensor(metrics_dict[key])).item()
            self.writer.add_scalar(f"{key}_{subset_str}", metric, epoch)
            print(f'{key}_{subset_str}: {metric}')
        print()

    def _critic_train_iteration(self, coarse, fine):

        fake = self.G(coarse)

        c_real = self.C(fine)
        c_fake = self.C(fake)

        gradient_penalty = hp.gp_lambda * self._gp(fine, fake, self.C)

        # Zero the gradients
        self.C_optimizer.zero_grad()

        c_real_mean = torch.mean(c_real)
        c_fake_mean = torch.mean(c_fake)

        critic_loss = c_fake_mean - c_real_mean + gradient_penalty
        w_estimate = c_real_mean - c_fake_mean

        critic_loss.backward(retain_graph = True)

        # Update the critic
        self.C_optimizer.step()

    def _generator_train_iteration(self, coarse, fine):

        self.G_optimizer.zero_grad()

        fake = self.G(coarse)
        c_fake = self.C(fake)

        # 
        g_loss = -torch.mean(c_fake)*hp.gamma

        # content loss
        g_loss += hp.content_lambda * nn.L1Loss()(fake, fine)

        g_loss.backward()
        self.G_optimizer.step()

    def _gp(self, real, fake, critic):

        current_batch_size = real.size(0)

        # Calculate interpolation
        alpha = torch.rand(current_batch_size, 1, 1, 1, requires_grad=True, device=hp.device)
        alpha = alpha.expand_as(real)

        interpolated = alpha * real.data + (1 - alpha) * fake.data

        # Calculate probability of interpolated examples
        critic_interpolated = critic(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(critic_interpolated.size(), device=hp.device),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(hp.batch_size, -1).to(hp.device)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return hp.gp_lambda * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, train_dl, val_dl, epoch):

        print(80*"=")
        train_metrics = self._initialize_metric_dicts()

        for i, data in enumerate(train_dl):
            coarse = data[0].to(hp.device)
            fine = data[1].to(hp.device)

            self._critic_train_iteration(coarse, fine)

            if self.num_steps%hp.critic_iterations == 0:
                self._generator_train_iteration(coarse, fine)
                
            # Track train set metrics
            train_metrics = self._calculate_metrics(
                self.G,
                self.C,
                coarse,
                fine,
                train_metrics,
            )

            self.num_steps += 1
            break
        
        # mean of all batches and log to file
        with torch.no_grad():
            # train_metrics = self._post_epoch_metric_mean(train_metrics, "train")
            self._tensorboard_metrics_mean(train_metrics, "train", epoch)

            val_metrics = self._initialize_metric_dicts()

            for data in val_dl:
                coarse = data[0].to(hp.device)
                fine = data[1].to(hp.device)

                # Track train set metrics
                val_metrics = self._calculate_metrics(
                    self.G,
                    self.C,
                    coarse,
                    fine,
                    val_metrics,
                )
                break

            # Take mean of all batches and log to file
            # val_metrics = self._post_epoch_metric_mean(val_metrics, "val")
            self._tensorboard_metrics_mean(val_metrics, "val", epoch)

        if epoch % hp.save_every == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(self.G.state_dict(), f"models/G_{epoch}.pth")
            torch.save(self.C.state_dict(), f"models/C_{epoch}.pth")
            print(f"Models saved at epoch {epoch}")
        
        self.writer.flush()
        print(f'epoch: {epoch}/{hp.epochs}')

    def train(self, train_dl, val_dl):
        self.num_steps = 0
        for epoch in range(hp.epochs):
            self._train_epoch(train_dl, val_dl, epoch)
