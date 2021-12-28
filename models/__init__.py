# -*- encoding: utf-8 -*-
'''
@Time       : 12/28/21 10:35 AM
@Author     : Jiang.xx
@Email      : cxyth@live.com
@Description: 构建模型的入口
'''
from .unetplusplus import UnetPlusPlus


def create_model(type, arch, encoder, in_channel, out_channel, pretrained=None):
    if type == 'smp':
        import segmentation_models_pytorch as smp
        smp_net = getattr(smp, arch)
        model = smp_net(               # smp.UnetPlusPlus
            encoder_name=encoder,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=pretrained,     # use `imagenet` pretrained weights for encoder initialization
            in_channels=in_channel,     # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=out_channel,     # model output channels (number of classes in your dataset)
        )
    elif type == 'custom':
        # 自定义模型
        assert arch == 'UnetPlusPlus'
        model = UnetPlusPlus(  # UnetPlusPlus
            encoder_name=encoder,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=pretrained,  # use `imagenet` pretrained weights for encoder initialization
            in_channels=in_channel,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=out_channel,  # model output channels (number of classes in your dataset)
        )
    return model
