# -*- encoding: utf-8 -*-
'''
@Time       : 12/28/21 10:35 AM
@Author     : Jiang.xx
@Email      : cxyth@live.com
@Description: 分析数据集常用代码
'''
import os
import cv2
import random 
import shutil
import numpy as np
from tqdm import tqdm
import os.path as osp
# import palettable
import matplotlib.pyplot as plt
from collections import defaultdict

Class2Id = {
    'background': 0,
    'budiling': 1,
    'road': 2,
    'forest': 3,
    'glass': 4,
    'farmland': 5,
    'water': 6,
    'bareland': 7,
    # 'mine': 8,
    # 'special': 9
}

Id2Class = {
    0: 'background',
    1: 'budiling',
    2: 'road',
    3: 'forest',
    4: 'glass',
    5: 'farmland',
    6: 'water',
    7: 'bareland',
    # 8: 'mine',
    # 9: 'special'
}


colors = [  # bgr color
    [0, 0, 0], # 'background',
    [0, 255, 255],  # 'budiling',
    [0, 128, 0],  # 'road',
    [0, 255, 0],  # 'forest',
    [0, 128, 0],  # 'glass',
    [0, 0, 255],  # 'farmland',
    [255, 0, 0],  # 'water',
    [128, 128, 0],  # 'bareland',
    # [128, 0, 128],  # 'mine',
    # [255, 255, 0],  # 'special',
]


def percentage_truncation(im_data, lower_percent=0.001, higher_percent=99.999, per_channel=True):
    '''
        将uint 16bit转换成uint 8bit (压缩法)
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


def randering_mask(image, mask, n_label, colors, alpha=0.6, beta=0.4):
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
    # mask = mask_to_onehot(mask, num_classes=n_label)  # shape=(h*w,n_label),即长度为h*w的one-hot向量
    # mask = mask.reshape((-1, n_label))
    mask = np.matmul(mask, colors)  # (h*w,n_label) x (n_label,3) ——> (h*w,3)
    mask = mask.reshape((mh, mw, 3)).astype(np.uint8)
    return cv2.addWeighted(image, alpha, mask, beta, 0)


def get_fid(dir, ext):
    files = os.listdir(dir)
    fids = []
    for f in files:
        if f.endswith(ext):
            fids.append(os.path.splitext(f)[0])
    return fids


def unique_folder(data_dir):
    fids = get_fid(data_dir, '.tif')
    fnum = len(fids)
    n_class = len(Class2Id.items())
    counts = np.zeros(n_class)
    for fid in tqdm(fids):
        label = cv2.imread(osp.join(data_dir, fid+'.tif'), cv2.IMREAD_GRAYSCALE)
        _count = np.bincount(label.flatten(), minlength=n_class)
        counts += _count

    total = np.sum(counts)
    assert total == fnum*256*256

    ratios = counts / total

    print('{:<10} | {:<10}'.format('class', 'ratio(%)'))
    for i in range(n_class):
        print('{:<10} | {:<10f}'.format(Id2Class[i], ratios[i]*100))


def compute_mean_std(data_dirs):

    paths = []
    for _dir in data_dirs:
        fids = get_fid(_dir, '.tif')
        paths.extend([osp.join(_dir, fid+'.tif') for fid in fids])
    fnum = len(paths)
    print("samples:", len(paths))
    mean = np.zeros(4)
    std = np.zeros(4)
    for path in tqdm(paths):
        # print('\nimage>', path)

        img = cv2.imread(path, cv2.IMREAD_LOAD_GDAL)
        img = img.astype(np.float32) / 255.

        # for i in range(4):
            # print('val>', img[..., i].min(), img[..., i].max())
        for i in range(4):
            _mean = img[..., i].mean()
            # print('mean>', _mean)
            mean[i] += _mean

        for i in range(4):
            _std = img[..., i].std()
            # print('std>', _std)
            std[i] += img[..., i].std()

    print('mean:', mean / fnum)
    print('std:', std / fnum)
    return mean / fnum, std / fnum


def compute_mean_std_2(data_dirs):

    paths = []
    for _dir in data_dirs:
        fids = get_fid(_dir, '.tif')
        paths.extend([osp.join(_dir, fid+'.tif') for fid in fids])
    fnum = len(paths)
    print("samples:", len(paths))

    images = []
    for p in tqdm(paths):
        img = cv2.imread(p, cv2.IMREAD_LOAD_GDAL)
        img = img[:, :, :, np.newaxis]
        images.append(img)
    images = np.concatenate(images, axis=3).astype(np.float32) / 255. 
    
    means, stdevs = [], []
    for i in tqdm(range(4)):
        pixels = images[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    print('mean:', means)
    print('std:', stdevs)



def plot_sample_proportion(dataset_dir, save_dir):
    all_files = [osp.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.tif')]
    total_num = len(all_files)

    data = np.zeros(10)
    for img_path in tqdm(all_files):
        label = cv2.imread(img_path.replace('.tif', '.png'), cv2.IMREAD_GRAYSCALE)
        vals, counts = np.unique(label, return_counts=True)
        for i, v in enumerate(vals):
            assert v != 0
            data[v-1] += counts[i]
     
    dpi = 96
    plt.figure(figsize=(1280/dpi, 1280/dpi), dpi=dpi)
    patches, texts, autotexts = plt.pie(x=data,                                    
            labels=list(Class2Id.keys()),
            # colors=palettable.cartocolors.qualitative.Bold_9.mpl_colors,
            autopct='%.2f%%',
        )
    # 添加图例
    plt.legend(patches, 
            list(Class2Id.keys()),
            loc="lower left",
            bbox_to_anchor=(0, 0, 0.5, 1),
        )
    # plt.show()
    plt.savefig(osp.join(save_dir, 'sample_proportion.png'))



def sample_proportion(dataset_dirs):

    def softmax(x):
        # 计算每行的最大值
        row_max = x.max()
        # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
        row_max=row_max.reshape(-1, 1)
        x = x - row_max
        # 计算e的指数次幂
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp)
        s = x_exp / x_sum
        return s

    def logxy(x, y):
        x = x * np.ones_like(y, dtype=y.dtype)
        return np.log(y) / np.log(x)

    n_class = 10
    label_files = []
    for url in dataset_dirs:
        label_files.extend(sorted([os.path.join(url, f) for f in os.listdir(url) if f.endswith('.png')]))
    total_num = len(label_files)

    class_count = np.zeros(n_class, dtype=np.float64)
    for label_path in tqdm(label_files):
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = label - 1
        _count = np.bincount(label.flatten(), minlength=n_class)
        class_count += _count
    # base_prob = 1 / (class_count / sum(class_count))
    base_prob = np.sum(class_count) / class_count
    print(base_prob)
    # 数值太大，使用log压缩值范围，否则softmax会溢出
    base_prob = np.log2(base_prob)
    print(base_prob)
    base_prob = softmax(base_prob)
    print(base_prob)

    # 使用各类别的base_prob计算样本的weights
    sampling_weights = []
    for label_path in tqdm(label_files):
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = label - 1
        _count = np.bincount(label.flatten(), minlength=n_class)
        _prob = _count * base_prob
        _prob = np.sum(_prob) / label.size
        sampling_weights.append(_prob)
        # if _prob > 0.1: print(label_path)
    sampling_weights = np.array(sampling_weights)
    print(sampling_weights.max())
    print(np.sum(sampling_weights > 0.001))
    print(np.sum(sampling_weights > 0.005))
    print(np.sum(sampling_weights > 0.01))
    print(np.sum(sampling_weights > 0.05))
    print(np.sum(sampling_weights > 0.1))




def check_class(dataset_dir, checkout_dir):
    # 对每一类建立文件夹并可视化label，便于检查分析原始样本
    classlist = list(Class2Id.keys())

    def clustering(all_files, cls_names=[]):
        # 分别收集各个类别的样本
        class_sets = defaultdict(list)
        for img_path in tqdm(all_files):
            label = cv2.imread(img_path.replace('/images/', '/labels/'), cv2.IMREAD_GRAYSCALE)
            vals, counts = np.unique(label, return_counts=True)
            for name in cls_names:
                cls_id = Class2Id[name]
                if cls_id in vals:
                    class_sets[name].append(img_path)
        return class_sets

    _path = osp.join(dataset_dir, 'images')
    all_data = [osp.join(_path, f) for f in os.listdir(_path) if f.endswith('.tif')]
    total_num = len(all_data)
    class_sets = clustering(all_data, classlist)

    print('> check:')
    for cname in classlist:
        check_class_dir = osp.join(checkout_dir, cname)
        os.makedirs(check_class_dir, exist_ok=True)
        for f in tqdm(class_sets[cname]):
            img = cv2.imread(f)
            label = cv2.imread(f.replace('/images/', '/labels/'), cv2.IMREAD_GRAYSCALE)
            cls_id = Class2Id[cname]
            cls_mask = np.where(label == cls_id, cls_id, 0)
            oths_cls_mask = np.where(label == cls_id, 0, label)
            n_class = len(Class2Id.items())
            img_cls_mask = randering_mask(img, cls_mask, n_class, colors, alpha=0.5, beta=0.5)
            img_oths_cls_mask = randering_mask(img, oths_cls_mask, n_class, colors, alpha=0.5, beta=0.5)
            # _img = percentage_truncation(img, lower_percent=1, higher_percent=99)
            res = np.concatenate([img, img_cls_mask, img_oths_cls_mask], axis=1)
            cv2.imwrite(osp.join(check_class_dir, osp.split(f)[-1]), res)

    print('total samples:', total_num)
    for cname in classlist:
        print('{:<20}'.format(cname), len(class_sets[cname]))


if __name__ == '__main__':
    src_dir = '/home/obtai/workspace/BuildingExtraction/DATASET/v4/256/train/labels'
    # unique_folder(src_dir)
    # class | ratio( % )
    # background | 0.228309
    # budiling | 8.412600
    # road | 3.976995
    # forest | 55.065185
    # glass | 2.834115
    # farmland | 14.184958
    # water | 15.269011
    # bareland | 0.028826


    # src_dirs = ['/home/ioai/Desktop/USER887/TIANCHI/dataset/src/suichang_round1_train_210120/',
    #             '/home/ioai/Desktop/USER887/TIANCHI/dataset/src/suichang_round1_test_partA_210120/']
    # compute_mean_std(src_dirs)
    # mean: [0.11894047 0.12947237 0.10506935 0.50785508]
    # std: [0.06690406 0.07723667 0.06991268 0.1627165 ]
    # compute_mean_std_2(src_dirs)
    # mean: [0.11894194, 0.12947349, 0.1050701, 0.50788707]
    # std: [0.08124223, 0.09198588, 0.08354711, 0.20507027]

    # dataset = '/home/work/ioai/USER887/TIANCHI2021/dataset/src/suichang_round1_train_210120/'
    dataset = '/home/obtai/workspace/BuildingExtraction/DATASET/v4/256/train'
    out_dir = '/home/obtai/workspace/BuildingExtraction/DATASET/v4/256/train_check/'
    check_class(dataset, out_dir)
    # plot_sample_proportion(dataset, out_dir)

    # dataset_dirs = ['/home/work/ioai/USER887/TIANCHI2021/dataset/suichang_round1_train_210120/',
    #                 '/home/work/ioai/USER887/TIANCHI2021/dataset/suichang_round2_train_210316/']
    # sample_proportion(dataset_dirs)
