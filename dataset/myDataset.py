# -*- encoding: utf-8 -*-
import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, Sampler
import albumentations as A
from albumentations.pytorch import ToTensorV2


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class myDataset(Dataset):

    def __init__(self, dataset_urls, transform, img_ext='.tif'):
        self.dataset_urls = dataset_urls
        self.transform = transform
        self.dataset = self._load_dataset(img_ext)
        print(f'> Creating dataset with {len(self.dataset)} samples.')

    def _load_dataset(self, img_ext):
        dataset = []
        for url in self.dataset_urls:
            img_dir = os.path.join(url, 'images')
            label_dir = os.path.join(url, 'labels')
            fids = sorted([f for f in os.listdir(img_dir) if f.endswith(img_ext)])
            dataset.extend([(os.path.join(img_dir, fid), os.path.join(label_dir, fid)) for fid in fids])
        return dataset

    def get_dataset(self):
        return self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        data = self.dataset[i]
        image = cv2.cvtColor(cv2.imread(data[0]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(data[1], cv2.IMREAD_GRAYSCALE)
        # mask = (mask > 0).astype(np.uint8)
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        mask = mask[None, :, :]
        return {
            'image': image,
            'label': mask.long()
        }


def get_sample_weights(dataset_urls, n_class):
    def softmax(x):
        # 计算每行的最大值
        row_max = x.max()
        # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
        row_max = row_max.reshape(-1, 1)
        x = x - row_max
        # 计算e的指数次幂
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp)
        s = x_exp / x_sum
        return s

    label_files = []
    for url in dataset_urls:
        _dir = os.path.join(url, 'labels')
        label_files.extend(sorted([os.path.join(_dir, f) for f in os.listdir(_dir) if f.endswith('.tif')]))

    print("> Apply resampling, analysis of the label：")
    class_count = np.zeros(n_class, dtype=np.float64)
    for label_path in tqdm(label_files):
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        _count = np.bincount(label.flatten(), minlength=n_class)
        class_count += _count
    # base_prob = 1 / (class_count / sum(class_count))
    base_prob = np.sum(class_count) / class_count
    # print('base_prob 1', base_prob)
    # 数值太大，使用log压缩值范围，否则softmax会溢出
    base_prob = np.log(base_prob)
    base_prob = softmax(base_prob)
    # print('base_prob 2', base_prob)

    # 使用各类别的base_prob计算样本的weights
    sampling_weights = []
    for label_path in tqdm(label_files):
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        _count = np.bincount(label.flatten(), minlength=n_class)
        _prob = _count * base_prob
        _prob = np.sum(_prob) / label.size
        sampling_weights.append(_prob)
    return np.array(sampling_weights, dtype=np.float32)


class batchtest_Dataset(Dataset):
    def __init__(self, rs_imagery, tile_size, overlap, transform):
        self.image = rs_imagery
        self.tile_size = tile_size
        self.overlap = overlap
        self.transform = transform
        self.tile_boxes = self._get_tile_coordinates_boxes(rs_imagery)

    def _get_tile_coordinates_boxes(self, image):
        overlap = self.overlap
        tile_size = self.tile_size
        stride = tile_size - overlap
        img_h, img_w, img_c = image.shape

        n_h = int(np.ceil((img_h - tile_size) / stride)) + 1
        n_w = int(np.ceil((img_w - tile_size) / stride)) + 1

        all_box = []
        for i in range(n_h):
            dh = min(i * stride, img_h - tile_size)
            for j in range(n_w):
                dw = min(j * stride, img_w - tile_size)
                all_box.append([dh, dh + tile_size, dw, dw + tile_size])
        return all_box

    def __len__(self):
        return len(self.tile_boxes)

    def __getitem__(self, i):
        box = self.tile_boxes[i]
        y1, y2, x1, x2 = box
        tile = self.image[y1:y2, x1:x2]
        transformed = self.transform(image=tile)
        tile = transformed['image']

        return {
            'image': tile,
            'box': box
        }


def get_train_transform(input_size):
    return A.Compose([
                A.Resize(input_size, input_size),
                A.OneOf([
                    A.Flip(),
                    A.RandomRotate90(),
                ], p=1.0),
                A.OneOf([
                    A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2),
                    A.ElasticTransform(alpha=0, sigma=0, alpha_affine=20, border_mode=0),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, border_mode=0),
                ], p=0.5),
                A.GaussNoise(p=0.3),
                A.RandomGamma(p=0.3),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0, p=0.5),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ])


def get_val_transform(input_size):
    return A.Compose([
                A.Resize(input_size, input_size),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ])


if __name__ == '__main__':
    import os
    import cv2
    from torch.utils.data import DataLoader
    '''test DataGenerator'''

    dirs = ["../../DATASET/v1/ccf/train"]
    class_info = be_Class2Id
    transform = get_train_transform()
    BS = 4
    D = batchtest_Dataset(dirs, class_info, transform, target_class='water')
    dataset = D.get_dataset()

    for i in range(10):
        print(dataset[i])

    print(20*'=')

    test_data = batchtest_Dataset(img, tile_size, overlap, transform)
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=batch_size,
        drop_last=False)

    data_loader = DataLoader(dataset=D, batch_sampler=S, num_workers=1)

    mean = np.array([0.11894194, 0.12947349, 0.1050701])
    std = np.array([0.08124223, 0.09198588, 0.08354711])

    for idx, batch_samples in enumerate(data_loader):
        imgs, labels = batch_samples['image'], batch_samples['label']
        imgs = imgs.numpy()
        labels = labels.numpy()
        print(imgs.shape, labels.shape)

        imgs = np.transpose(imgs, (0, 2, 3, 1))
        labels = np.transpose(labels, (0, 2, 3, 1))
        for i in range(BS):
            img = imgs[i, :, :, 0:3]
            img = (img * std + mean) * 255
            img = img.astype(np.uint8)
            label = labels[i].astype(np.uint8) * 255
            cv2.imwrite(os.path.join('test', 'sample_b{:03d}_{:02d}.tif'.format(idx, i)), img)
            cv2.imwrite(os.path.join('test', 'sample_b{:03d}_{:02d}.png'.format(idx, i)), label)

        if idx >= 10: break
        