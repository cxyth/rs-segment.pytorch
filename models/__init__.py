# -*- encoding: utf-8 -*-
'''
@Time       : 12/28/21 10:35 AM
@Author     : Jiang.xx
@Email      : cxyth@live.com
@Description: 构建模型的入口
'''
from models.segformer.segformer import SegFormer
# from segformer.segformer import SegFormer
from .fcsiam import FCSiamConc, FCSiamDiff
from .mySiamNet import mySiamDiffUnet

custom_models = {
    'segformer': SegFormer,
}


def create_model(cfg: dict):
    model_type = cfg['type']
    arch = cfg['arch']
    if model_type == 'smp':
        import segmentation_models_pytorch as smp
        smp_net = getattr(smp, arch)

        encoder = cfg.get('encoder', 'resnet34')
        pretrained = cfg.get('pretrained', 'imagenet')
        in_channel = cfg.get('in_channel', 3)
        out_channel = cfg.get('out_channel', 2)
        aux_params = cfg.get('aux_params', None)

        model = smp_net(               # smp.UnetPlusPlus
            encoder_name=encoder,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=pretrained,     # use `imagenet` pretrained weights for encoder initialization
            in_channels=in_channel,     # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=out_channel,     # model output channels (number of classes in your dataset)
            aux_params=aux_params
        )
    elif model_type == 'siamese':
        # 自定义模型
        archs = [FCSiamConc, FCSiamDiff, mySiamDiffUnet]
        archs_dict = {a.__name__.lower(): a for a in archs}
        try:
            model_class = archs_dict[arch.lower()]
        except KeyError:
            raise KeyError("Wrong architecture type `{}`. Available options are: {}".format(
                arch, list(archs_dict.keys()),
            ))
        encoder = cfg.get('encoder', 'resnet34')
        pretrained = cfg.get('pretrained', 'imagenet')
        in_channel = cfg.get('in_channel', 3)
        out_channel = cfg.get('out_channel', 2)
        aux_params = cfg.get('aux_params', None)
        return model_class(
            encoder_name=encoder,
            encoder_weights=pretrained,
            in_channels=in_channel,
            classes=out_channel,
            aux_params=aux_params,
        )
    elif model_type == 'custom':
        # 自定义模型
        assert arch.lower() in custom_models.keys()
        net = custom_models[arch.lower()]
        model = net(
            backbone=cfg['backbone'],
            decode_head=cfg['seg_head'],
            pretrained=cfg.get('pretrained', None)
        )
    else:
        print('type error')
        exit()
    return model


if __name__ == '__main__':
    from torchsummary import summary

    backbone = {'type': 'mit_b1'}
    seg_head = {
        'type': 'SegFormerHead',
        'in_channels': [32, 64, 160, 256],  # b1
        'feature_strides': [4, 8, 16, 32],
        'channels': 128,
        'dropout_ratio': 0.1,
        'num_classes': 9,
        'decoder_params': {'embed_dim': 256}  # b1
    }
    pretrained = '../pretrained/mit_b1.pth'

    model = SegFormer(backbone=backbone, decode_head=seg_head, pretrained=pretrained).cuda()

    for name, param in model.named_parameters(recurse=False):
        print(name)

    for child_name, child_mod in model.named_children():
        print('child:', child_name)
        print(child_mod)
