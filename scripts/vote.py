# -*- encoding: utf-8 -*-
'''
@Time       : 12/28/21 10:35 AM
@Author     : Jiang.xx
@Email      : cxyth@live.com
@Description: 投票
'''
import sys
sys.path.append('../')
import cv2
import os
from utils import mask_to_onehot
from tqdm import tqdm
import os.path as osp
import numpy as np


def get_fid(dir, ext):
    return [f for f in os.listdir(dir) if f.endswith(ext)]


def vote(results, weights, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    fids0 = get_fid(results[0], '.png')
    fids1 = get_fid(results[1], '.png')
    fids2 = get_fid(results[2], '.png')

    assert len(fids0)==len(fids1) and len(fids1)==len(fids2)

    for f in tqdm(fids0):
        mask0 = cv2.imread(osp.join(results[0], f), cv2.IMREAD_GRAYSCALE)
        mask1 = cv2.imread(osp.join(results[1], f), cv2.IMREAD_GRAYSCALE)
        mask2 = cv2.imread(osp.join(results[2], f), cv2.IMREAD_GRAYSCALE)

        mask0 = mask_to_onehot(mask0 - 1, num_classes=10)
        mask1 = mask_to_onehot(mask1 - 1, num_classes=10)
        mask2 = mask_to_onehot(mask2 - 1, num_classes=10)

        _mask = mask0 + mask1 + mask2
        _mask = _mask + weights
        mask = np.argmax(_mask, axis=2)

        cv2.imwrite(osp.join(output_dir, f), mask+1)




if __name__ == '__main__':
    # self.classes = {1: 'farmland',
    #                 2: 'forest',
    #                 3: 'grass',
    #                 4: 'road',
    #                 5: 'urban_area',
    #                 6: 'countryside',
    #                 7: 'industrial_land',
    #                 8: 'construction',
    #                 9: 'water',
    #                 10: 'bareland'}

    class_weights = np.array([0.7, 0.9, 0.1, 0.3, 0.4, 0.5, 0.6, 0.2, 0.8, 0.0])
    results_dir = ['../tmp/dpl.v2p0.1/S1/predict_naked/',
                   '../tmp/dpl.v2p1.1/S1/predict_naked/',
                   '../tmp/dpl.v2p9.1/S1/predict_tta/']
    output_dir = './results'

    vote(results_dir, class_weights, output_dir)

