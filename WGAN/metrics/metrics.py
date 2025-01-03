import torch
import torchvision.models as models

from pytorch_msssim import ssim
from WGAN.metrics.perceptual_VGG import PerceptualVGG


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

def VGG_loss(HR: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    device = HR.device

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    HR = (HR - mean) / std
    fake = (fake - mean) / std

    perceptual_model = PerceptualVGG(layers=['relu_1', 'relu_2', 'relu_3']).to(device)

    HR_vgg = perceptual_model(HR)
    fake_vgg = perceptual_model(fake)

    loss = 0.0
    for key in HR_vgg.keys():
        loss += torch.mean((HR_vgg[key] - fake_vgg[key]) ** 2)

    return loss
