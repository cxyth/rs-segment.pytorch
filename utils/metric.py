# -*- encoding: utf-8 -*-
'''
@Time       : 12/28/21 10:35 AM
@Author     : Jiang.xx
@Email      : cxyth@live.com
@Description: 精度评估方法
'''
import os
import time
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi


class Metric(object):
    """
    Class to calculate mean-iou using fast_hist method
    """
    def __init__(self, num_class, binary=False):
        self.num_class = num_class
        self.binary = binary
        self.hist = np.zeros((num_class, num_class))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_class)
        hist = np.bincount(
            self.num_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_class ** 2).reshape(self.num_class, self.num_class)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def reset(self):
        self.hist = np.zeros((self.num_class, self.num_class))

    def get_confusion_matrix(self):
        return self.hist

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        precisions = np.diag(self.hist) / self.hist.sum(axis=0)
        recalls = np.diag(self.hist) / self.hist.sum(axis=1)
        ious = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        # kappa
        po = acc
        pe = np.sum(self.hist.sum(axis=0) * self.hist.sum(axis=1)) / self.hist.sum()**2
        kappa = (po - pe) / (1 - pe)
        if not self.binary:
            mean_precision = np.nanmean(precisions)
            mean_recall = np.nanmean(recalls)
            miou = np.nanmean(ious)
            freq = self.hist.sum(axis=1) / self.hist.sum()
            fwavacc = (freq[freq > 0] * ious[freq > 0]).sum()
            return {
                'acc': acc,
                'fwavacc': fwavacc,
                'class_precision': precisions,
                'mean_precision': mean_precision,
                'class_recall': recalls,
                'mean_recall': mean_recall,
                'class_iou': ious,
                'mean_iou': miou,
                'kappa': kappa
            }
        else:
            return {
                'acc': acc,
                'precision': precisions[1],
                'recall': recalls[1],
                'iou': ious[1],
                'kappa': kappa
            }


# =================================================================================
#               像素级评估指标（语义分割）
# =================================================================================
import prettytable as pt
from scipy import sparse

EPS = 1e-7

def get_console_file_logger(name, level, logdir):
    logger = logging.Logger(name)
    logger.setLevel(level=level)
    logger.handlers = []
    BASIC_FORMAT = "%(asctime)s, %(levelname)s:%(name)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel(level=level)

    fhlr = logging.FileHandler(os.path.join(logdir, str(time.time()) + '.log'))
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)

    return logger


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self._total = sparse.coo_matrix((num_classes, num_classes), dtype=np.float32)

    def forward(self, y_true, y_pred):
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()

        y_pred = y_pred.reshape((-1,))
        y_true = y_true.reshape((-1,))

        v = np.ones_like(y_pred)
        cm = sparse.coo_matrix((v, (y_true, y_pred)), shape=(self.num_classes, self.num_classes), dtype=np.float32)
        self._total += cm

        return cm

    @property
    def dense_cm(self):
        return self._total.toarray()

    @property
    def sparse_cm(self):
        return self._total

    def reset(self):
        num_classes = self.num_classes
        self._total = sparse.coo_matrix((num_classes, num_classes), dtype=np.float32)

    @staticmethod
    def plot(confusion_matrix):
        return NotImplementedError


class PixelMetric(ConfusionMatrix):
    def __init__(self, num_classes, logdir=None, logger=None, class_names=None):
        super(PixelMetric, self).__init__(num_classes)
        if logdir is not None:
            os.makedirs(logdir, exist_ok=True)
        self.logdir = logdir
        if logdir is not None and logger is None:
            self._logger = get_console_file_logger('PixelMetric', logging.INFO, self.logdir)
        elif logger is not None:
            self._logger = logger
        else:
            self._logger = None
        self._class_names = class_names
        if class_names:
            assert num_classes == len(class_names)

    @property
    def logger(self):
        return self._logger

    @staticmethod
    def compute_iou_per_class(confusion_matrix):
        """
        Args:
            confusion_matrix: numpy array [num_classes, num_classes] row - gt, col - pred
        Returns:
            iou_per_class: float32 [num_classes, ]
        """
        sum_over_row = np.sum(confusion_matrix, axis=0)
        sum_over_col = np.sum(confusion_matrix, axis=1)
        diag = np.diag(confusion_matrix)
        denominator = sum_over_row + sum_over_col - diag

        iou_per_class = diag / (denominator + EPS)

        return iou_per_class

    @staticmethod
    def compute_recall_per_class(confusion_matrix):
        sum_over_row = np.sum(confusion_matrix, axis=1)
        diag = np.diag(confusion_matrix)
        recall_per_class = diag / (sum_over_row + EPS)
        return recall_per_class

    @staticmethod
    def compute_precision_per_class(confusion_matrix):
        sum_over_col = np.sum(confusion_matrix, axis=0)
        diag = np.diag(confusion_matrix)
        precision_per_class = diag / (sum_over_col + EPS)
        return precision_per_class

    @staticmethod
    def compute_overall_accuracy(confusion_matrix):
        diag = np.diag(confusion_matrix)
        return np.sum(diag) / (np.sum(confusion_matrix) + EPS)

    @staticmethod
    def compute_F_measure_per_class(confusion_matrix, beta=1.0):
        precision_per_class = PixelMetric.compute_precision_per_class(confusion_matrix)
        recall_per_class = PixelMetric.compute_recall_per_class(confusion_matrix)
        F1_per_class = (1 + beta ** 2) * precision_per_class * recall_per_class / (
                (beta ** 2) * precision_per_class + recall_per_class + EPS)

        return F1_per_class

    @staticmethod
    def cohen_kappa_score(cm_th):
        cm_th = cm_th.astype(np.float32)
        n_classes = cm_th.shape[0]
        sum0 = cm_th.sum(axis=0)
        sum1 = cm_th.sum(axis=1)
        expected = np.outer(sum0, sum1) / (np.sum(sum0) + EPS)
        w_mat = np.ones([n_classes, n_classes])
        w_mat.flat[:: n_classes + 1] = 0
        k = np.sum(w_mat * cm_th) / (np.sum(w_mat * expected) + EPS)
        return 1. - k

    def _log_summary(self, table, dense_cm):
        if self.logger is not None:
            self.logger.info('\n' + table.get_string())
            if self.logdir is not None:
                np.save(os.path.join(self.logdir, 'confusion_matrix-{time}.npy'.format(time=time.time())), dense_cm)
        else:
            print(table)

    def summary_iou(self):
        dense_cm = self._total.toarray()
        iou_per_class = PixelMetric.compute_iou_per_class(dense_cm)
        miou = iou_per_class.mean()

        tb = pt.PrettyTable()
        tb.field_names = ['class', 'iou']
        for idx, iou in enumerate(iou_per_class):
            tb.add_row([idx, iou])
        tb.add_row(['mIoU', miou])

        self._log_summary(tb, dense_cm)

        return tb

    def summary_all(self, dec=5):
        dense_cm = self._total.toarray()

        iou_per_class = np.round(PixelMetric.compute_iou_per_class(dense_cm), dec)
        miou = np.round(iou_per_class.mean(), dec)
        F1_per_class = np.round(PixelMetric.compute_F_measure_per_class(dense_cm, beta=1.0), dec)
        mF1 = np.round(F1_per_class.mean(), dec)
        overall_accuracy = np.round(PixelMetric.compute_overall_accuracy(dense_cm), dec)
        kappa = np.round(PixelMetric.cohen_kappa_score(dense_cm), dec)

        precision_per_class = np.round(PixelMetric.compute_precision_per_class(dense_cm), dec)
        mprec = np.round(precision_per_class.mean(), dec)
        recall_per_class = np.round(PixelMetric.compute_recall_per_class(dense_cm), dec)
        mrecall = np.round(recall_per_class.mean(), dec)

        tb = pt.PrettyTable()
        if self._class_names:
            tb.field_names = ['name', 'class', 'iou', 'f1', 'precision', 'recall']
            for idx, (iou, f1, precision, recall) in enumerate(
                    zip(iou_per_class, F1_per_class, precision_per_class, recall_per_class)):
                tb.add_row([self._class_names[idx], idx, iou, f1, precision, recall])

            tb.add_row(['', 'mean', miou, mF1, mprec, mrecall])
            tb.add_row(['', 'OA', overall_accuracy, '-', '-', '-'])
            tb.add_row(['', 'Kappa', kappa, '-', '-', '-'])

        else:
            tb.field_names = ['class', 'iou', 'f1', 'precision', 'recall']
            for idx, (iou, f1, precision, recall) in enumerate(
                    zip(iou_per_class, F1_per_class, precision_per_class, recall_per_class)):
                tb.add_row([idx, iou, f1, precision, recall])

            tb.add_row(['mean', miou, mF1, mprec, mrecall])
            tb.add_row(['OA', overall_accuracy, '-', '-', '-'])
            tb.add_row(['Kappa', kappa, '-', '-', '-'])

        self._log_summary(tb, dense_cm)

        return tb


# =================================================================================
#               对象级评估指标（实例分割）
# =================================================================================

def instance_evaluate_binary(y_true, y_pred, iou_threshold=0.5):
    '''
        面向对象的评估指标，对于每个模型预测的目标区域目标，通过IoU评测真实的目标区域与模型输
        出的区域之间的重叠程度，当IoU>=iou_threshold时, 认为预测正确。在此基础上计算
        precision, recall, f1。该函数仅用于二分类mask。
    :param y_true: 实际标签, (H, W)的单类别mask。
    :type y_true: 0/1二值图
    :param y_pred: 预测结果，(H, W)的单类别mask。
    :type y_pred: 0/1二值图
    :param iou_threshold: iou阈值，真实目标与预测输出的重叠程度。
    :type iou_threshold: float
    :return: 对象级评估指数TP, FP, FN, precision, recall, f1
    :rtype: float
    '''
    # 给标签mask的每个连续预测区域标记不同的值，视为不同的实例（对象），并返回实例的数量
    true_marked, true_inst_num = ndi.label(y_true)
    # 统计标记后每个实例的代表值/所占像素数量
    true_inst_vals, true_val_counts = np.unique(true_marked, return_counts=True)
    true_flt = true_marked.flatten()

    # 给预测mask的每个连续预测区域标记不同的值，视为不同的实例（对象），并返回实例的数量
    pred_marked, pred_inst_num = ndi.label(y_pred)
    # 统计标记后每个实例的代表值/所占像素数量
    pred_inst_vals, pred_val_counts = np.unique(pred_marked, return_counts=True)
    pred_flt = pred_marked.flatten()

    assert y_true.max() <= 1 and y_pred.max() <= 1
    # 标签与预测结果相加后判断得到所有实例的交集，int_mask(intersection mask)
    int_mask = ((y_true + y_pred) > 1).astype(np.uint8)
    int_marked, int_inst_num = ndi.label(int_mask)
    # 统计标记后每个实例的代表值/所在位置索引/所占像素数量
    int_inst_vals, int_val_indices, int_val_counts = np.unique(int_marked, return_index=True, return_counts=True)

    TP = 0  # 真正例TP
    val_pair_list = []  # 存放每个交集的对应true/pred的实例标记值数对（当实例之间有多个分离交集，只保存一个）
    intersection_list = []  # 与val_pair_list等长，存放相交的实例对的所有交集的总像素量
    # 遍历所有交集区域，统计实例对和对应的交集
    for i in range(int_inst_vals.size):
        val = int_inst_vals[i]
        if val == 0:
            continue
        _intersection = int_val_counts[i]
        _index = int_val_indices[i]
        _true_val = true_flt[_index]
        _pred_val = pred_flt[_index]

        if [_true_val, _pred_val] not in val_pair_list:
            val_pair_list.append([_true_val, _pred_val])
            intersection_list.append(_intersection)
        else:
            intersection_list[val_pair_list.index([_true_val, _pred_val])] += _intersection
    # 遍历所有存在交集的实例对，统计预测正确的实例
    for i, (_true_val, _pred_val) in enumerate(val_pair_list):
        _intersection = intersection_list[i]
        _true_count = int(true_val_counts[true_inst_vals == _true_val])
        _pred_count = int(pred_val_counts[pred_inst_vals == _pred_val])

        _union = _true_count + _pred_count - _intersection
        _iou = _intersection / _union
        if _iou >= iou_threshold:
            TP += 1

    FP = pred_inst_num - TP
    FN = true_inst_num - TP
    precision = TP / pred_inst_num if pred_inst_num else 1
    recall = TP / true_inst_num if true_inst_num else 1
    f1 = 2. / ((1. / (recall)) + (1. / (precision))) if recall and precision else 0
    return TP, FP, FN, precision, recall, f1


def instance_evaluate_muticlass(y_true, y_pred, iou_thresholds):
    '''
        用于对象级评估模型输出的多分类mask(H, W, k)，类别数为 K，
        如果使用softmax作为输出且不关心背景类的话，需要去掉该通道。
    :param y_true: 实际标签, (H, W， K)的多分类形式的mask。
    :type y_true: 多类别mask
    :param y_pred: 预测结果，(H, W, K)的多分类的mask。
    :type y_pred: 多类别mask
    :param iou_thresholds: 可以是长度为(K)的list，单独指定每一类别的iou阈值。
    :type iou_thresholds: float/list
    :return: 多类别的对象级平均评估指数TP, FP, FN, Precision, Recall, F1
    :rtype: float
    '''
    assert y_true.ndim == 3 and y_pred.ndim == 3, 'y_true.shape {}. y_pred.shape {}'.format(y_true.shape, y_pred.shape)

    H, W, K = y_true.shape
    class_num = K
    if isinstance(iou_thresholds, list):
        iou_thresholds = np.array(iou_thresholds)
    else:
        iou_thresholds = np.ones(class_num) * iou_thresholds
    assert iou_thresholds.size == class_num

    TPs = np.zeros(class_num)
    FPs = np.zeros(class_num)
    FNs = np.zeros(class_num)
    Ps = np.zeros(class_num)
    Rs = np.zeros(class_num)
    F1s = np.zeros(class_num)
    for i in range(class_num):
        tp, fp, fn, precision, recall, f1 = instance_evaluate_binary(y_true[:, :, i], y_pred[:, :, i],
                                                                     iou_thresholds[i])
        TPs[i] = tp
        FPs[i] = fp
        FNs[i] = fn
        Ps[i] = precision
        Rs[i] = recall
        F1s[i] = f1
    return TPs, FPs, FNs, Ps, Rs, F1s


# =================================================================================
#               tools
# =================================================================================

def plot_confusion_matrix(cm, classes, savename, title='Confusion Matrix'):
    import itertools

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes, rotation=45)

    plt.axis("equal")

    ax = plt.gca()
    left, right = plt.xlim()
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

    thresh = float(cm.max() / 2.)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{}'.format(int(cm[i, j]))
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if float(num) > thresh else "black")

    plt.ylabel('Actual')
    plt.xlabel('Predict')

    plt.tight_layout()
    plt.savefig(savename, transparent=False, dpi=160)
    # plt.show()

