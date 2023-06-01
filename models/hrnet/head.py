import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

BN_MOMENTUM = 0.01
FINAL_CONV_KERNEL = 1


class FCNHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        assert isinstance(in_channels, (list, tuple))
        in_channels = np.int(np.sum(in_channels))
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_classes,
                kernel_size=FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if FINAL_CONV_KERNEL == 3 else 0)
        )

    def forward(self, inputs):
        x0, x1, x2, x3 = inputs      # len=4, 1/4,1/8,1/16,1/32

        x0_h, x0_w = x0.size(2), x0.size(3)
        x1 = F.interpolate(x1, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=(x0_h, x0_w), mode='bilinear', align_corners=True)

        x = torch.cat([x0, x1, x2, x3], 1)
        x = self.last_layer(x)

        return x
