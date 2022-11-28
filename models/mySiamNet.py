from typing import Any, Callable, Optional, Sequence, Union

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.base.model import SegmentationModel
from torch import Tensor

Unet.__module__ = "segmentation_models_pytorch"

class mySiamDiffUnet(Unet):  # type: ignore[misc]
    """Base on Fully-convolutional Siamese Difference (FC-Siam-diff).

    If you use this model in your research, please cite the following paper:

    * https://doi.org/10.1109/ICIP.2018.8451652
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a new FCSiamConc model.

        Args:
            encoder_name: Name of the classification model that will
                be used as an encoder (a.k.a backbone) to extract features
                of different spatial resolution
            encoder_depth: A number of stages used in encoder in range [3, 5].
                two times smaller in spatial dimensions than previous one
                (e.g. for depth 0 we will have features. Each stage generate
                features with shapes [(N, C, H, W),], for depth
                1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on). Default is 5
            encoder_weights: One of **None** (random initialization), **"imagenet"**
                (pre-training on ImageNet) and other pretrained weights (see table
                with available weights for each encoder_name)
            decoder_channels: List of integers which specify **in_channels**
                parameter for convolutions used in decoder. Length of the list
                should be the same as **encoder_depth**
            decoder_use_batchnorm: If **True**, BatchNorm2d layer between
                Conv2D and Activation layers is used. If **"inplace"** InplaceABN
                will be used, allows to decrease memory consumption. Available
                options are **True, False, "inplace"**
            decoder_attention_type: Attention module used in decoder of the model.
                Available options are **None** and **scse**. SCSE paper
                https://arxiv.org/abs/1808.08127
            in_channels: A number of input channels for the model,
                default is 3 (RGB images)
            classes: A number of classes for output mask (or you can think as a number
                of channels of output mask)
            activation: An activation function to apply after the final convolution
                n layer. Available options are **"sigmoid"**, **"softmax"**,
                **"logsoftmax"**, **"tanh"**, **"identity"**, **callable**
                and **None**. Default is **None**
        """
        kwargs["aux_params"] = None
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: input images of shape (b, t, c, h, w)

        Returns:
            predicted change masks of size (b, classes, h, w)
        """
        x1 = x[:, 0]
        x2 = x[:, 1]
        features1, features2 = self.encoder(x1), self.encoder(x2)
        features = [features2[i] - features1[i] for i in range(len(features1))]
        decoder_output = self.decoder(*features)
        masks: Tensor = self.segmentation_head(decoder_output)
        return masks
