# -*- encoding: utf-8 -*-
'''
@Time       : 12/28/21 10:35 AM
@Author     : Jiang.xx
@Email      : cxyth@live.com
@Description: 图像处理
'''
import os
import cv2
import numpy as np
import math
import time


def apply_colormap(image, heatmap, alpha=0.5, beta=0.5):
    '''
        渲染mask至image上
    :param image: 渲染的底图 (h*w*c)
    :type image: numpy
    :param heatmap: 所要绘制的热度图 (h*w)
    :type heatmap: numpy(0~1 float)
    :return: opencv图像
    :rtype: opencv image
    '''
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, alpha, heatmap, beta, 0)


def randering_mask(image, mask, n_label, colors, alpha=0.5, beta=0.5):
    '''
        渲染mask至image上
    :param image: 渲染的底图 (h*w*c)
    :type image: numpy
    :param mask: 所要渲染的二值图 (h*w)
    :type mask: numpy
    :param n_label: 标签种类数
    :type n_label: int
    :param colors: 颜色矩阵 exp:三个种类则[[255,0,255],[0,255,0],[255,0,0]]
    :type colors: numpy or list
    :return: opencv图像
    :rtype: opencv image
    '''
    colors = np.array(colors)
    mh, mw = mask.shape
    mask = np.eye(n_label)[mask.reshape(-1)]    # shape=(h*w, n_label),即长度为h*w的one-hot向量
    mask = np.matmul(mask, colors)  # (h*w,n_label) x (n_label,3) ——> (h*w,3)
    mask = mask.reshape((mh, mw, 3)).astype(np.uint8)
    return cv2.addWeighted(image, alpha, mask, beta, 0)


def percentage_truncation(im_data, lower_percent=0.001, higher_percent=99.999, per_channel=True):
    '''
    :param im_data: 图像矩阵(h, w, c)
    :type im_data: numpy
    :param lower_percent: np.percentile的最低百分位
    :type lower_percent: float
    :param higher_percent: np.percentile的最高百分位
    :type higher_percent: float
    :return: 返回图像矩阵(h, w, c)
    :rtype: numpy
    '''
    if per_channel:
        out = np.zeros_like(im_data, dtype=np.uint8)
        for i in range(im_data.shape[2]):
            a = 0  # np.min(band)
            b = 255  # np.max(band)
            c = np.percentile(im_data[:, :, i], lower_percent)
            d = np.percentile(im_data[:, :, i], higher_percent)
            if (d - c) == 0:
                out[:, :, i] = im_data[:, :, i]
            else:
                t = a + (im_data[:, :, i] - c) * (b - a) / (d - c)
                t = np.clip(t, a, b)
                out[:, :, i] = t
    else:
        a = 0  # np.min(band)
        b = 255  # np.max(band)
        c = np.percentile(im_data, lower_percent)
        d = np.percentile(im_data, higher_percent)
        out = a + (im_data - c) * (b - a) / (d - c)
        out = np.clip(out, a, b).astype(np.uint8)
    return out
