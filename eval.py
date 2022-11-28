# -*- coding: utf-8 -*-
import cv2
from tqdm import tqdm
import os
import numpy as np
from utils import Metric, PixelMetric


def eval(pred_dir, label_dir, output_dir):
    img_set = [f for f in os.listdir(pred_dir) if f[-4:] in ['.png', '.tif']]
    cls_names = ['building_change']

    metric_op = PixelMetric(2, output_dir)

    tbar = tqdm(img_set)
    tbar.set_description('Evaluate')
    for fid in tbar:
        # tbar.set_postfix_str(fid)
        pred_path = os.path.join(pred_dir, fid)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        pred = (pred > 0).astype(np.uint8)

        label_path = os.path.join(label_dir, fid)
        if os.path.isfile(label_path):
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            label = (label > 0).astype(np.uint8)
        else:
            raise FileNotFoundError(f'file [{label_path}] not found.')

        y_true = label.ravel()
        y_pred = pred.ravel()
        metric_op.forward(y_true, y_pred)

    metric_op.summary_all()


if __name__ == '__main__':
    label_dir = r'E:\Jiang\changedetection\datasets\S2Looking\test\label'
    pred_dir = './runs/unet_efb0_abs.s2_512.4/test_final/results'
    output_dir = './runs/unet_efb0_abs.s2_512.4/test_final/eval'
    os.makedirs(output_dir, exist_ok=True)
    eval(pred_dir, label_dir, output_dir)
