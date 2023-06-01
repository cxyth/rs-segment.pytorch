# -*- encoding: utf-8 -*-
'''
@Time       : 12/28/21 10:35 AM
@Author     : Jiang.xx
@Email      : cxyth@live.com
@Description: 用于自定义数据集的pytorch Dataset示例
'''
import os
import cv2
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self, im_data, tile_size, overlap, transform, channel_first=False):
        if channel_first:       # 需要把(C, H, W)转为(H, W, C)
            im_data = np.transpose(im_data, (1, 2, 0))
        self.image = im_data
        self.tile_size = tile_size
        self.overlap = overlap
        self.transform = transform
        self.tile_coords = self._get_tile_coordinates(im_data)

    def _get_tile_coordinates(self, image):
        stride = self.tile_size - self.overlap
        img_h, img_w, img_c = image.shape
        n_h = int(np.ceil((img_h - self.tile_size) / stride)) + 1
        n_w = int(np.ceil((img_w - self.tile_size) / stride)) + 1
        windows = []
        for i in range(n_h):
            dh = min(i * stride, img_h - self.tile_size)
            for j in range(n_w):
                dw = min(j * stride, img_w - self.tile_size)
                if np.sum(image[dh:dh + self.tile_size, dw:dw + self.tile_size, :]) == 0:
                    continue
                windows.append([dh, dh + self.tile_size, dw, dw + self.tile_size])
        return windows

    def __len__(self):
        return len(self.tile_coords)

    def __getitem__(self, i):
        window = self.tile_coords[i]
        y1, y2, x1, x2 = window
        tile = self.image[y1:y2, x1:x2]
        transformed = self.transform(image=tile)
        tile = transformed['image']

        return {
            'image': tile,
            'window': np.array(window)
        }


class Sentinel2Dataset(Dataset):

    def __init__(self, im_data, tile_size, overlap, transform, channel_first=False):
        if channel_first:       # 需要把(C, H, W)转为(H, W, C)
            im_data = np.transpose(im_data, (1, 2, 0))
        self.image = im_data
        self.tile_size = tile_size
        self.overlap = overlap
        self.transform = transform
        self.tile_coords = self._get_tile_coordinates(im_data)

    def _get_tile_coordinates(self, image):
        stride = self.tile_size - self.overlap
        img_h, img_w, img_c = image.shape
        n_h = int(np.ceil((img_h - self.tile_size) / stride)) + 1
        n_w = int(np.ceil((img_w - self.tile_size) / stride)) + 1
        windows = []
        for i in range(n_h):
            dh = min(i * stride, img_h - self.tile_size)
            for j in range(n_w):
                dw = min(j * stride, img_w - self.tile_size)
                if np.sum(image[dh:dh + self.tile_size, dw:dw + self.tile_size, :]) == 0:
                    continue
                windows.append([dh, dh + self.tile_size, dw, dw + self.tile_size])
        return windows

    def __len__(self):
        return len(self.tile_coords)

    def __getitem__(self, i):
        window = self.tile_coords[i]
        y1, y2, x1, x2 = window
        im_data = self.image[y1:y2, x1:x2]
        im_data = self.transform(im_data)
        return {
            'image': torch.from_numpy(im_data).to(torch.float32),
            'window': np.array(window)
        }
        