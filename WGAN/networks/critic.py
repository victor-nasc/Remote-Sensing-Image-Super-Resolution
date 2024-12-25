import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, input_channels=3, features=64):
        super(Critic, self).__init__()
        self.features = features

        # feature extraction
        self.backbone = nn.Sequential(
            self.conv_block(input_channels, features),             # 3x128x128  -> 64x128x128
            self.conv_block(features, features, stride=2),         # 64x128x128 -> 64x64x64
            self.conv_block(features, features * 2),               # 64x64x64   -> 128x64x64
            self.conv_block(features * 2, features * 2, stride=2), # 128x64x64  -> 128x32x32
            self.conv_block(features * 2, features * 4),           # 128x32x32  -> 256x32x32
            self.conv_block(features * 4, features * 4, stride=2), # 256x32x32  -> 256x16x16
            self.conv_block(features * 4, features * 8),           # 256x16x16  -> 512x16x16
            self.conv_block(features * 8, features * 8, stride=2), # 512x16x16  -> 512x8x8
        )

        # classifier
        self.head = nn.Sequential(
            nn.Linear(8 * features * 8 * 8, 100),  # Flattened size
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(100, 1),
        )

    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input)
        out = torch.flatten(out, 1)  
        out = self.head(out)

        return out

