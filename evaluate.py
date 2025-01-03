import csv
import time
import torch
import torch.nn as nn
import WGAN.config.hyperparams as hp

from torch.utils.data import DataLoader
from WGAN.processing.dataloader import SRDataset
from WGAN.networks.generator import Generator
from WGAN.metrics.metrics import SSIM, PSNR


METRICS_TO_CALCULATE = {
    "MAE": nn.L1Loss(),
    "MSE": nn.MSELoss(),
    "SSIM": SSIM,
    "PSNR": PSNR, 
}

experiment = 'cn10_01_crit5'

def write_csv(image_metrics):
    with open(f'metrics_per_image_{experiment}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        
        header = ['Image'] + list(METRICS_TO_CALCULATE.keys())
        writer.writerow(header)
        
        for image_idx, metrics in enumerate(image_metrics):
            row = [image_idx + 1] + [metrics[key] for key in METRICS_TO_CALCULATE.keys()]
            writer.writerow(row)

def evaluate(G, LR, HR):
    with torch.no_grad():
        fake = G(LR).detach()

        image_metrics = []
        for i in range(LR.size(0)):  # for each image in the batch
            metrics = {}
            for key in METRICS_TO_CALCULATE.keys():
                metrics[key] = METRICS_TO_CALCULATE[key](HR[i:i+1], fake[i:i+1]).detach().cpu().item()
            image_metrics.append(metrics)

        return image_metrics

def main():
    device = torch.device("cuda:0")

    test_ds = SRDataset('data/test')
    test_dl = DataLoader(test_ds, batch_size=hp.batch_size, shuffle=False, num_workers=4)

    generator = Generator().to(device)

    checkpoints = f'/home/victornasc/Remote-Sensing-Image-Super-Resolution/models_{experiment}/'
    generator.load_state_dict(torch.load(checkpoints + 'G_400.pth'))
    
    image_metrics = []  

    start_time = time.time()
    for batch_idx, data in enumerate(test_dl):
        LR = data[0].to(device)
        HR = data[1].to(device)
        
        batch_image_metrics = evaluate(generator, LR, HR)
        image_metrics.extend(batch_image_metrics)
        
        print(f'Batch {batch_idx + 1} / {len(test_dl)}', end='\r')
    
    write_csv(image_metrics)
    print(f'Time: {time.time() - start_time}')

if __name__ == '__main__':
    main()
