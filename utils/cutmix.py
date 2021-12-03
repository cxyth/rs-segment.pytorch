import numpy as np
import torch
import torch.nn as nn


def cutmix(batch, beta):
    data, targets = batch['image'], batch['label']
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(beta, beta)

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))
    
    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    targets[:, :, y0:y1, x0:x1] = shuffled_targets[:, :, y0:y1, x0:x1]

    return {'image': data, 'label': targets}


class CutMixCollator:
    def __init__(self, cutmix_prob, beta):
        self.prob = cutmix_prob
        self.beta = beta

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        if np.random.rand() < self.prob:
            batch = cutmix(batch, self.beta)
        return batch


