# -*- encoding: utf-8 -*-
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .transform import RandomDiscreteScale

levir_class_info = {
    'building': 1
}

# rgb  & by imagenet
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def random_swap(img1, img2):
    if np.random.random() > 0.5:
        _img = img1.copy()
        img1 = img2.copy()
        img2 = _img.copy()
    return img1, img2


class levir_dataset(Dataset):
    def __init__(self, dataset_url, transform, mode):
        assert mode in ['train', 'val', 'test']
        self.dataset_url = dataset_url
        self.mode = mode
        self.transform = transform
        self.dataset = self._load_dataset()
        print(f'> Creating dataset with {len(self.dataset)} examples.')

    def _load_dataset(self):
        dataset = []
        fids = sorted([f for f in os.listdir(os.path.join(self.dataset_url, 'label')) if f[-4:]=='.png'])
        for f in fids:
            img1 = os.path.join(self.dataset_url, 'A', f)
            assert os.path.isfile(img1), img1
            img2 = os.path.join(self.dataset_url, 'B', f)
            assert os.path.isfile(img2), img2
            label = os.path.join(self.dataset_url, 'label', f)
            assert os.path.isfile(label), label
            dataset.append((img1, img2, label))
        return dataset

    def get_dataset(self):
        return self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        (img1_path, img2_path, label_path) = self.dataset[i]
        fname = os.path.split(img1_path)[-1]

        img1 = cv2.imread(img1_path)
        img1 = np.ascontiguousarray(img1[:, :, ::-1])
        img2 = cv2.imread(img2_path)
        img2 = np.ascontiguousarray(img2[:, :, ::-1])
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = (label > 0).astype(np.uint8)

        if self.mode == 'train':
            # img1, img2, label = random_crop(img1, img2, label, patch_sz=512)
            # img1, img2 = random_swap(img1, img2)
            sample = self.transform(image=img1, image2=img2, mask=label)
            img1, img2, mask = sample['image'], sample['image2'], sample['mask']
            imgs = torch.cat([img1.unsqueeze_(0), img2.unsqueeze_(0)], dim=0)
            mask = mask[None, :, :]
            return {
                'image': imgs,
                'label': mask.long(),
            }
        elif self.mode == 'val':
            sample = self.transform(image=img1, image2=img2, mask=label)
            img1, img2, mask = sample['image'], sample['image2'], sample['mask']
            imgs = torch.cat([img1.unsqueeze_(0), img2.unsqueeze_(0)], dim=0)
            mask = mask[None, :, :]
            return {
                'image': imgs,
                'label': mask.long(),
            }
        else:
            sample = self.transform(image=img1, image2=img2)
            img1, img2 = sample['image'], sample['image2']
            imgs = torch.cat([img1.unsqueeze_(0), img2.unsqueeze_(0)], dim=0)
            return {
                'image': imgs,
                'fname': fname,
            }


def get_train_transform():
    return A.Compose(
        [
            A.OneOf([
                A.HorizontalFlip(True),
                A.VerticalFlip(True),
                A.RandomRotate90(True),
            ], p=0.75),
            # A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0, p=0.5),
            # A.RandomResizedCrop(512, 512, scale=(0.25, 2.25), ratio=(0.75, 1.333), always_apply=True),
            # RandomDiscreteScale([0.75, 1.25, 1.5], p=0.5),
            # A.RandomCrop(512, 512, True),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        additional_targets={'image2': 'image'}
    )


def get_val_transform():
    return A.Compose(
        [
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        additional_targets={'image2': 'image'}
    )