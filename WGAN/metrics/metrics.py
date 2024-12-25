import torch
import torch.nn as nn

from pytorch_msssim import ssim
from torcheval.metrics import PeakSignalNoiseRatio 


def wass_loss(HR, fake):
    return HR - fake

def normalize_image(img: torch.Tensor) -> torch.Tensor:
    img_min, img_max = img.min(), img.max()
    img = (img - img_min) / (img_max - img_min + 1e-8)

    assert float(img.max()) <= 1.0 and float(img.min()) >= 0.0
    return img

def SSIM(HR: torch.Tensor, fake: torch.Tensor) -> float:
    HR = normalize_image(HR)
    fake = normalize_image(fake)

    return ssim(HR, fake, data_range=1.0)

def PSNR(HR: torch.Tensor, fake: torch.Tensor) -> float:
    HR = normalize_image(HR)
    fake = normalize_image(fake)

    psnr = PeakSignalNoiseRatio(data_range=1.0)
    psnr.update(HR, fake)

    return psnr.compute()

