# -*- encoding: utf-8 -*-
'''
@Time       : 12/28/21 10:35 AM
@Author     : Jiang.xx
@Email      : cxyth@live.com
@Description: 语义分割相关的工具集
'''
import cv2
import random
import colorsys
import numpy as np
from tqdm import tqdm
from skimage import segmentation, measure, morphology, color
from scipy import ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt


def apply_watershed(prob_map, seed_thr, mask):
    '''
        以分水岭算法为基础的后处理，将语义分割的概率图转换为实例分割并解决部分粘连的情况。
    :param prob_map: (H, W), 值为0~1的概率图。
    :type prob_map:
    :param seed_thr: 一个阈值，概率图中大于该值的像素作为分水岭的种子。
    :type seed_thr: float
    :param mask: 分水岭的 mask。
    :type mask:
    :return: labels 分水岭操作后的实例分割图像/ markers 标记好的种子图像
    :rtype:
    '''

    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=1) # 腐蚀
    # mask = cv2.dilate(mask, kernel, iterations=1) # 膨胀
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # 开运算，先腐蚀再膨胀，有助于消除噪音
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # 闭运算，先膨胀后腐蚀，消除前景对象内的小孔
    # mask = cv2.dilate(mask, kernel, iterations=1) # 膨胀

    seeds = (prob_map > seed_thr).astype(np.uint8)
    seeds, _ = ndi.label(seeds)

    prob_reverse = (1. - prob_map)
    labels = morphology.watershed(prob_reverse, seeds, mask=mask, watershed_line=True)

    return labels, seeds


def mask_remove_large_objects(mask, area_threshold):
    '''
        用于清除mask中面积大于阈值的块
    :param mask: 实际标签, (H, W)的单类别mask。
    :type mask: 0/1二值图
    :param area_threshold: 碎片像素面积的阈值，大于或等于该值的块将被丢弃。
    :type area_threshold: float
    :return: new_mask
    :rtype: 0/1二值图
    '''

    # ar: 待操作的bool型数组。
    # min_size: 最小连通区域尺寸，小于该尺寸的都将被删除。默认为64.
    # connectivity: 邻接模式，1表示4邻接，2表示8邻接
    # in_place: bool型值，如果为True,表示直接在输入图像中删除小块区域，否则进行复制后再删除。默认为False.
    large_objects = morphology.remove_small_objects(ar=mask, min_size=area_threshold, connectivity=1, in_place=False)

    return mask - large_objects


def mask_remove_small_holds(mask, area_threshold):
    '''
        用于清除模型输出的单分类mask的细小碎片
    :param mask: 实际标签, (H, W)的单类别mask。
    :type mask: 0/1二值图
    :param area_threshold: 碎片像素面积的阈值，小于该值的碎片将被丢弃。
    :type area_threshold: float
    :return: new_mask
    :rtype: 0/1二值图
    '''

    # ar: 待操作的bool型数组。
    # min_size: 最小连通区域尺寸，小于该尺寸的都将被删除。默认为64.
    # connectivity: 邻接模式，1表示4邻接，2表示8邻接
    # in_place: bool型值，如果为True,表示直接在输入图像中删除小块区域，否则进行复制后再删除。默认为False.
    labeled = measure.label(mask, connectivity=1)
    new_mask = morphology.remove_small_holes(ar=labeled, area_threshold=area_threshold, connectivity=1)
    new_mask = (new_mask > 0).astype(np.uint8)
    return new_mask


def mask_remove_small_objects(mask, area_threshold):
    '''
        用于清除模型输出的单分类mask的细小碎片
    :param mask: 实际标签, (H, W)的单类别mask。
    :type mask: 0/1二值图
    :param area_threshold: 碎片像素面积的阈值，小于该值的碎片将被丢弃。
    :type area_threshold: float
    :return: new_mask
    :rtype: 0/1二值图
    '''

    # ar: 待操作的bool型数组。
    # min_size: 最小连通区域尺寸，小于该尺寸的都将被删除。默认为64.
    # connectivity: 邻接模式，1表示4邻接，2表示8邻接
    # in_place: bool型值，如果为True,表示直接在输入图像中删除小块区域，否则进行复制后再删除。默认为False.
    labeled = measure.label(mask, connectivity=1)
    new_mask = morphology.remove_small_objects(ar=labeled, min_size=area_threshold, connectivity=1)
    new_mask = (new_mask > 0).astype(np.uint8)
    return new_mask


def mask_remove_small_objects_multiclasse(masks, area_thresholds):
    '''
        用于清除模型输出的多分类mask(H, W, k)的细小碎片,默认通道0为背景，类别数为 K-1，
        可选择返回处理时间。
    :param masks: 实际标签, (H, W)的单类别mask。
    :type masks: 0/1二值图
    :param area_threshold: 碎片像素面积的阈值，小于该值的碎片将被丢弃。
                           可以是长度为(K-1)的list，单独指定每一类别的阈值。
    :type area_threshold: float/list
    :return: new_mask
    :rtype: 0/1二值图
    '''
    assert masks.ndim == 3, print('check input masks,expect 3-dimension array, got {}.'.format(masks.ndim))

    # 默认0通道为背景, 不过滤背景mask
    H, W, K = masks.shape
    class_num = K - 1
    if isinstance(area_thresholds, list):
        area_thresholds = np.array(area_thresholds)
    else:
        area_thresholds = np.ones(class_num) * area_thresholds
    assert area_thresholds.size == class_num

    new_masks = np.zeros_like(masks)
    for i in range(class_num):
        new_masks[:, :, i + 1] = mask_remove_small_objects(masks[:, :, i + 1], area_thresholds[i])

    _back_ground = np.zeros((H, W))
    _back_ground[np.sum(new_masks, axis=-1) == 0] = 1
    new_masks[:, :, 0] = _back_ground

    return new_masks


def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (H,W,K) where the last dim is a one
    hot encoding vector
    """
    H, W = mask.shape
    _onehot = np.eye(num_classes)[mask.reshape(-1)]    # shape=(h*w, n_label),即长度为h*w的one-hot向量
    _onehot = _onehot.reshape(H, W, num_classes)
    return _onehot


def img_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.

    eg.
        for mask (shape = [H, W, 1]):
            palette = [[0], [128], [255]]
            gt_onehot = mask_to_onehot(gt, palette)     # shape = [H, W, 3]

        for colormap (shape = [H, W, 3]):
            palette = [[0, 0, 0], [192, 224, 224], [128, 128, 64], [0, 192, 128], [128, 128, 192], [128, 128, 0]]
            gt_onehot = mask_to_onehot(gt, palette)     # shape = [H, W, 6]
    """
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map


def onehot_to_mask(mask):
    """
    Converts a mask (H,W,K) to (H,W)
    """
    _mask = np.argmax(mask, axis=-1)
    # _mask[_mask != 0] += 1
    return _mask


def onehot_to_colormap(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    return x


def mask_to_binary_edges(mask, radius=2):
    """
    Converts a segmentation mask (H,W) to a binary edgemap (H,W)
    """
    if radius < 0:
        return mask
    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    dist = distance_transform_edt(mask_pad) + distance_transform_edt(1.0 - mask_pad)
    dist = dist[1:-1, 1:-1]
    dist[dist > radius] = 0
    edgemap = dist
    edgemap = np.expand_dims(edgemap, axis=2)
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation

    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.

    """
    def close_contour(contour):
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        return contour

    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def sementic_splash(image, mask, n_label, colors=None, alpha=0.5, beta=0.5):
    '''
        推理结果可视化，将预测的mask绘制到原图
    :param image: 原图 (h*w*c)
    :type image: numpy
    :param mask: 要绘制的mask (h*w)
    :type mask: numpy
    :param n_label: 标签种类数
    :type n_label: int
    :param colors: 颜色列表 eg.三个种类则[[255,0,255],[0,255,0],[255,0,0]]
    :type colors: numpy or list
    :return: opencv图像
    :rtype: opencv image
    '''
    if colors is not None:
        colors = np.array(colors)
    else:
        colors = random_colors(n_label)
        colors = np.array(colors) * 255
    mh, mw = mask.shape
    mask = np.eye(n_label)[mask.reshape(-1)]  # shape=(h*w, n_label),即长度为h*w的one-hot向量
    mask = np.matmul(mask, colors)  # (h*w,n_label) x (n_label,3) ——> (h*w,3)
    mask = mask.reshape((mh, mw, 3)).astype(np.uint8)
    return cv2.addWeighted(image, alpha, mask, beta, 0)


def instance_splash(image, masks, onehot=True, colors=None, alpha=0.5):
    """
    masks: can be mask with shape[height, width] or one-hot mask with shape[height, width, num_instances]
    colors: (optional) An array or colors to use with each object
    """
    # Number of instances
    if onehot:
        N = masks.shape[-1]
    else:
        N = masks.max()

        # Generate random colors
    colors = colors or random_colors(N)
    masked_image = image.astype(np.uint32).copy()

    for i in tqdm(range(N)):
        color = colors[i]
        # Mask
        if onehot:
            mask = masks[:, :, i]
        else:
            mask = (masks == i + 1).astype(np.uint8)

        # Apply the given mask to the image.
        for c in range(3):
            masked_image[:, :, c] = np.where(mask == 1, masked_image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                             masked_image[:, :, c])
    return masked_image.astype(np.uint8)


def splash_instances_to_image_cv2(image, mask, colors=None, alpha=0.4):
    '''
        将分割结果标记为单个实例并赋予不同颜色， 背景值默认为0，颜色为(0, 0, 0)。
    :param image: 原始图像。
    :type image: RGB/BGR/GRAY/None，0~255
    :param mask: 实际标签, (H, W)的单类别mask。
    :type masks: 0/1二值图
    :return: masked_image
    :rtype: 渲染后的图像，输入的image会被转为灰度图。
    '''
    insts_map, _ = ndi.label(mask)
    # color.label2rgb(label, image=None, colors=None, alpha=0.3, bg_label=-1, bg_color=(0, 0, 0), image_alpha=1, kind='overlay')
    masked_image = color.label2rgb(insts_map, image, colors, alpha, bg_label=0, bg_color=(0,0,0), image_alpha=1)
    masked_image = (masked_image * 255).astype(np.uint8)
    return masked_image