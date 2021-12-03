import os
import cv2
import json
import random 
import shutil
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os.path as osp

class_name = {1: '耕地',
              2: '林地',
              3: '草地',
              4: '道路',
              5: '城镇建设用地',
              6: '农村建设用地',
              7: '工业用地',
              8: '构筑物',
              9: '水域',
              10: '裸地'}


class_ids = {
    'farmland': 1,
    'forest': 2,
    'grass': 3,
    'road': 4,
    'urban_area': 5,
    'countryside': 6,
    'industrial_land': 7,
    'construction': 8,
    'water': 9,
    'bareland': 10
}

bgr_colors = [
    [0, 0, 0], # 'background',
    [128, 128, 0], # 'farmland',
    [0, 128, 0], # 'forest',
    [0, 255, 0], # 'grass',
    [255, 0, 255], # 'road',
    [0, 0, 255], # 'urban_area',
    [255, 0, 0], # 'countryside',
    [128, 0, 0], # 'industrial_land',
    [128, 0, 128], # 'construction',
    [255, 255, 0],# 'water',
    [0, 255, 255]# 'bareland'
]


def get_fid(dir, ext):
    files = os.listdir(dir)
    fids = []
    for f in files:
        if f.endswith(ext):
            fids.append(os.path.splitext(f)[0])
    return fids


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


def make_exp1(src_dir, dst_dir):
    fids = [osp.splitext(f)[0] for f in os.listdir(src_dir) if f.endswith('.tif')]
    random.seed(10101)
    random.shuffle(fids)
    total_num = len(fids)
    sub_num = total_num // 10
    train_num = sub_num * 9
    train_set = fids[:train_num]
    val_set = fids[train_num:]

    print('train num: {}  val num: {}'.format(len(train_set), len(val_set)))

    out_train_dir = osp.join(dst_dir, 'train')
    out_val_dir = osp.join(dst_dir, 'val')
    os.makedirs(osp.join(out_train_dir), exist_ok=True)
    os.makedirs(osp.join(out_val_dir), exist_ok=True)

    for fid in tqdm(train_set):
        image = fid+'.tif'
        shutil.copy(osp.join(src_dir, image), osp.join(out_train_dir, image))
        label = fid+'.png'
        shutil.copy(osp.join(src_dir, label), osp.join(out_train_dir, label))
    for fid in tqdm(val_set):
        image = fid+'.tif'
        shutil.copy(osp.join(src_dir, image), osp.join(out_val_dir, image))
        label = fid+'.png'
        shutil.copy(osp.join(src_dir, label), osp.join(out_val_dir, label))
    print('done!')


def make_exp2(src_dirs, dst_dir):
    fpaths = []
    for dir in src_dirs:
        fpaths.extend([osp.join(dir, f) for f in os.listdir(dir) if f.endswith('.tif')])
    random.seed(10101)
    random.shuffle(fpaths)
    total_num = len(fpaths)
    sub_num = total_num // 10
    train_num = sub_num * 9
    train_set = fpaths[:train_num]
    val_set = fpaths[train_num:]

    print('train num: {}  val num: {}'.format(len(train_set), len(val_set)))

    out_train_dir = osp.join(dst_dir, 'train')
    out_val_dir = osp.join(dst_dir, 'val')
    os.makedirs(osp.join(out_train_dir), exist_ok=True)
    os.makedirs(osp.join(out_val_dir), exist_ok=True)

    for fpath in tqdm(train_set):
        roundn = (fpath.split('/')[-2]).split('_')[1]
        src_dir, fname = osp.split(fpath)
        fid = osp.splitext(fname)[0]
        image = fid+'.tif'
        shutil.copy(osp.join(src_dir, image), osp.join(out_train_dir, roundn + '_' + image))
        label = fid+'.png'
        shutil.copy(osp.join(src_dir, label), osp.join(out_train_dir, roundn + '_' + label))
    for fpath in tqdm(val_set):
        roundn = (fpath.split('/')[-2]).split('_')[1]
        src_dir, fname = osp.split(fpath)
        fid = osp.splitext(fname)[0]
        image = fid+'.tif'
        shutil.copy(osp.join(src_dir, image), osp.join(out_val_dir, roundn + '_' + image))
        label = fid+'.png'
        shutil.copy(osp.join(src_dir, label), osp.join(out_val_dir, roundn + '_' + label))
    print('done!')



def make_v2(src_dir, json_path):
    fids = get_fid(src_dir, '.tif')
    total_num = len(fids)
    random.seed(10101)
    random.shuffle(fids)
    DATAs = [osp.join(src_dir, x+'.tif') for x in fids]

    data = {'data': DATAs}
    with open(json_path, 'w') as f:
         json.dump(data, f)
    print('done!')


def collect_class(all_files, class_names=[]):
    class_sets = defaultdict(list)
    for img_path in tqdm(all_files):
        label = cv2.imread(img_path.replace('.tif', '.png'), cv2.IMREAD_GRAYSCALE)
        for classname in class_names:
            cls_id = class_ids[classname]
            if np.any(label==cls_id):
                class_sets[classname].append(img_path)
    return class_sets

def make_v3(src_dir, dst_dir, split=10):
    classlist = ['farmland',
            'forest',
            'grass',
            'road',
            'urban_area',
            'countryside',
            'industrial_land',
            'construction',
            'water',
            'bareland']

    fids = get_fid(src_dir, '.tif')
    total_num = len(fids)
    sub_num = total_num // split
    random.seed(10101)
    random.shuffle(fids)

    all_data = [osp.join(src_dir, x+'.tif') for x in fids]
    
    _sets = collect_class(all_data, classlist)
    _data = {'class': class_ids, 'train_data':_sets, 'val_data': []}
    _json_path = osp.join(dst_dir, 'v3.json')
    with open(_json_path, 'w') as f:
         json.dump(_data, f)

    print('> check:')
    check_dir = osp.join(dst_dir, 'v3_check')
    for cname in classlist:
        check_class_dir = osp.join(check_dir, cname)
        os.makedirs(check_class_dir, exist_ok=True)
        for f in tqdm(_sets[cname]):
            img = cv2.imread(f, cv2.IMREAD_LOAD_GDAL)
            img = img[..., 0:3]
            label = cv2.imread(f.replace('.tif', '.png'), cv2.IMREAD_GRAYSCALE)
            cls_id = class_ids[cname]
            cls_mask = np.where(label==cls_id, cls_id, 0)
            oths_cls_mask = np.where(label==cls_id, 0, label)
            img_cls_mask = randering_mask(img, cls_mask, 11, bgr_colors, alpha=0.5, beta=0.5)
            img_oths_cls_mask = randering_mask(img, oths_cls_mask, 11, bgr_colors, alpha=0.5, beta=0.5)
            _img = percentage_truncation(img, lower_percent=0.1, higher_percent=99.9)
            res = np.concatenate([img, _img, img_cls_mask, img_oths_cls_mask], axis=1)
            cv2.imwrite(osp.join(check_class_dir, osp.split(f)[-1]), res)

    for i in range(split):
        _train = []
        _val = []
        val_start = i * sub_num
        val_stop = (i+1)*sub_num if i < (split-1) else total_num
        _train.extend(all_data[:val_start])
        _train.extend(all_data[val_stop:])
        _val.extend(all_data[val_start:val_stop])

        _sets = collect_class(_train, classlist)
        _data = {'class': class_ids, 'train_data':_sets, 'val_data': _val}
        _json_path = osp.join(dst_dir, 'v3p{}.json'.format(i))
        with open(_json_path, 'w') as f:
            json.dump(_data, f)

    print('done!')
    
        

if __name__ == '__main__':
    # src_dir = r'E:\TIANCHI2021\dataset\suichang_round1_train_210120'
    # dst_dir = r'E:\TIANCHI2021\dataset\v3'
    # make_v3(src_dir, dst_dir=dst_dir)

    # src_dir = '/home/work/ioai/USER887/TIANCHI2021/dataset/src/suichang_round1_train_210120'
    # dst_dir = '/home/work/ioai/USER887/TIANCHI2021/dataset/exp_r1val9'
    # make_exp1(src_dir, dst_dir)

    src_dirs = [
        '/home/work/ioai/USER887/TIANCHI2021/dataset/suichang_round1_train_210120',
        '/home/work/ioai/USER887/TIANCHI2021/dataset/suichang_round2_train_210316']
    dst_dir = '/home/work/ioai/USER887/TIANCHI2021/dataset/exp2_r1r2val9'
    make_exp2(src_dirs, dst_dir)
