import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SRDataset(Dataset):
    def __init__(self, dir):
        self.hr_dir = dir
        self.hr_imgs = sorted(os.listdir(self.hr_dir))
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.scale_factor = 4


    def __len__(self):
        return len(self.hr_imgs)

    def __getitem__(self, idx):
        hr_img = Image.open(os.path.join(self.hr_dir, self.hr_imgs[idx]))
        lr_img = self._downscale(hr_img)

        return self.transform(lr_img), self.transform(hr_img)
    
    def _downscale(self, img):
        return img.resize((img.width // self.scale_factor, img.height // self.scale_factor), Image.BICUBIC)