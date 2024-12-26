import torch
from pytorch_msssim import ssim


def wass_loss(HR: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    return HR - fake

def normalize_image(img: torch.Tensor) -> torch.Tensor:
    img_min, img_max = img.min(), img.max()
    img = (img - img_min) / (img_max - img_min + 1e-8)

    assert float(img.max()) <= 1.0 and float(img.min()) >= 0.0
    return img

def SSIM(HR: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    HR = normalize_image(HR)
    fake = normalize_image(fake)

    return ssim(HR, fake, data_range=1.0)

def PSNR(HR: torch.Tensor, fake: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    HR = normalize_image(HR)
    fake = normalize_image(fake)

    mse = torch.mean((HR - fake) ** 2)
    if mse == 0:
        return float('inf')

    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))

    return psnr
