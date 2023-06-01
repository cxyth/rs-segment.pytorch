# -*- encoding: utf-8 -*-
'''
@Time       : 12/28/21 10:35 AM
@Author     : Jiang.xx
@Email      : cxyth@live.com
@Description: 评估
'''
import os
import time
import torch
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
from torchsummary import summary
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import load_config, init_logger
from utils.metric import Metric, PixelMetric
from datasets.ImgMaskDataset import DualDataset, get_val_transform
from models import create_model

"""
    eg. python eval.py -c unet_efb0.1 -w last.pt
"""


@torch.no_grad()
def run(cfg_name, weight):
    cfg = load_config(cfg_name, "configs")
    cfgN = cfg['network']
    cfgN['pretrained'] = None
    cfgD = cfg['dataset']
    n_class = len(cfgD['cls_info'])
    log_dir = osp.join(cfg['run_dir'], cfg_name)
    #  check weight file
    ckpt_path = osp.join(log_dir, 'ckpt', weight)
    if not osp.exists(ckpt_path):
        print(f'file {ckpt_path} not found.')
        return
    val_transform = get_val_transform()
    test_data = DualDataset(data_dirs=cfgD['test_dirs'], transform=val_transform)
    img_c, img_h, img_w = test_data.__getitem__(0)['image'].shape
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=False)
    # network
    model = create_model(cfg=cfgN).cuda()
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    summary(model, input_size=(img_c, img_h, img_w))

    logger = init_logger(osp.join(log_dir, f'eval_on_{osp.splitext(weight)[0]}.log'))
    metric = PixelMetric(n_class, logger=logger)
    tbar = tqdm(test_loader)
    tbar.set_description('Evaluate on Test set')
    for i, batch_samples in enumerate(tbar):
        imgs, label = batch_samples['image'], batch_samples['label']
        pred = model(imgs.cuda())
        pred = torch.softmax(pred, dim=1).cpu().numpy()
        pred = np.argmax(pred, axis=1)
        # pred = (pred.squeeze(1).cpu().numpy() > 0.5)
        y_true = label.numpy().ravel()
        y_pred = pred.ravel()
        metric.forward(y_true, y_pred)
    metric.summary_all()


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-c', '--config', type=str, required=True, help='Name of the config file.')
    argparser.add_argument('-w', '--weight', default='best.pt', type=str, help='Choice a weight file.')
    argparser.add_argument('-g', '--gpus', default='0', type=str, help='gpus')
    return argparser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    run(args.config, args.weight)


