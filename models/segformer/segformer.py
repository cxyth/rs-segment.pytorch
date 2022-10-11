import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from .head import SegFormerHead

BACKBONES = {
    'mit_b0': mit_b0,
    'mit_b1': mit_b1,
    'mit_b2': mit_b2,
    'mit_b3': mit_b3,
    'mit_b4': mit_b4,
    'mit_b5': mit_b5
}

class SegFormer(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 pretrained=None):
        super(SegFormer, self).__init__()

        # backbone
        _args = backbone.copy()
        backbone_type = _args.pop('type')
        self.backbone = BACKBONES[backbone_type](**_args)
        # head
        _args = decode_head.copy()
        head_type = _args.pop('type')
        assert head_type == 'SegFormerHead'
        self.decode_head = SegFormerHead(**_args)
        self.num_classes = self.decode_head.num_classes

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if pretrained is not None:
            print(f'load model from: {pretrained}')
            self.backbone.init_weights(pretrained=pretrained)
            # self.decode_head.init_weights()

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        x = self.backbone(inputs)
        x = self.decode_head(x)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

