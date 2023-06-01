# -*- encoding: utf-8 -*-
'''
@Time       : 12/28/21 10:35 AM
@Author     : Jiang.xx
@Email      : cxyth@live.com
@Description: 构建模型的入口
'''
from models.segformer import SegFormer
from models.hrnet import HRNet


custom_models = {
    'segformer': SegFormer,
    'hrnet': HRNet,
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
            classes=out_channel,     # model output channels (number of classes in your datasets)
            aux_params=aux_params
        )
    elif model_type == 'custom':
        assert arch.lower() in custom_models.keys()
        net = custom_models[arch.lower()]
        encoder = cfg.get('encoder', 'mit_b0')
        pretrained = cfg.get('pretrained', None)
        in_channel = cfg.get('in_channel', 3)
        out_channel = cfg.get('out_channel', 2)
        activation = cfg.get('activation', None)
        model = net(
            encoder_name=encoder,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=pretrained,  # use `imagenet` pretrained weights for encoder initialization
            in_channels=in_channel,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=out_channel,  # model output channels (number of classes in your datasets)
            activation=activation,
        )

    else:
        print('type error')
        exit()
    return model


if __name__ == '__main__':
    from torchsummary import summary

    # backbone = {'type': 'mit_b1'}
    # seg_head = {
    #     'type': 'SegFormerHead',
    #     'in_channels': [64, 128, 320, 512],  # b1
    #     'feature_strides': [4, 8, 16, 32],
    #     'channels': 128,
    #     'dropout_ratio': 0.1,
    #     'num_classes': 2,
    #     'decoder_params': {'embed_dim': 256}  # b1
    # }
    # pretrained = '../model_data/mit_b1.pth'
    # model = SegFormer(backbone=backbone, decode_head=seg_head, pretrained=pretrained).cuda()
    # summary(model, input_size=(3, 512, 512))

    # for name, param in model.named_parameters(recurse=False):
    #     print(name)
    #
    # for child_name, child_mod in model.named_children():
    #     print('child:', child_name)
    #     print(child_mod)

    from segformer import SegFormer

    encoder = 'mit_b1'
    pretrained = 'imagenet'
    in_channel = 3
    out_channel = 2
    activation = None
    model = SegFormer(
        encoder_name=encoder,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=pretrained,  # use `imagenet` pretrained weights for encoder initialization
        in_channels=in_channel,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=out_channel,  # model output channels (number of classes in your datasets)
        activation=activation,
    ).cuda()

    summary(model, input_size=(3, 512, 512))
