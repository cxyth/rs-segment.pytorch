from typing import Optional, Union, List

import torch
from torch import nn
import torch.nn.functional as F
from segmentation_models_pytorch.base.modules import Activation

from .backbone import get_backbone
from .head import FCNHead


class HRNet(nn.Module):
    def __init__(
            self,
            encoder_name: str = "hrnet_w18",
            encoder_weights: Optional[str] = None,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
    ):
        super().__init__()
        self.encoder = get_backbone(encoder_name, encoder_weights)
        self.decode_head = FCNHead(
            in_channels=self.encoder.channels,      # eg. hrnet_w18.channels: [18, 36, 72, 144]
            num_classes=classes)
        self.activation = Activation(activation)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape

        x = self.encoder(x)
        x = self.decode_head(x)

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        x = self.activation(x)
        return x


