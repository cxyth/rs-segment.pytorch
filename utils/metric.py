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
def segment_evaluate_binary(y_true, y_pred):
    '''
        面向像素的语义分割常用评估指标。该函数仅用于二分类mask。
    :param y_true: 实际标签, (H, W)的单类别mask。
    :type y_true: 0/1二值图
    :param y_pred: 预测结果，(H, W)的单类别mask。
    :type y_pred: 0/1二值图
    :return: 像素级评估指数TP, FP, FN, TN, acc, precision, recall, f1
    :rtype: float
    '''
    TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))

    accuracy = (TP + TN) / (TP + FP + FN + TN)
    precision = 1.0 if np.equal(TP + FP, 0) else TP / (TP + FP)
    recall = 1.0 if np.equal(TP + FN, 0) else TP / (TP + FN)
    f1 = 2. / ((1. / (recall)) + (1. / (precision))) if recall and precision else 0
    iou = 1.0 if np.equal(TP + FP + FN, 0) else TP / (TP + FP + FN)

    return TP, FP, FN, TN, accuracy, precision, recall, f1, iou


def segment_evaluate_muticlass(y_true, y_pred):
    '''
        用于像素级评估模型输出的多分类mask(H, W, k)，类别数为 K，
        如果使用softmax作为输出且不关心背景类的话，需要去掉该通道。
    :param y_true: 实际标签, (H, W， K)的多分类形式的mask。
    :type y_true: 多类别mask
    :param y_pred: 预测结果，(H, W, K)的多分类的mask。
    :type y_pred: 多类别mask
    :return: 多类别的对象级平均评估指数TP, FP, FN, TN, accuracy, Precision, Recall, F1, IoU
    :rtype: float
    '''
    assert y_true.ndim == 3 and y_pred.ndim == 3, 'y_true.shape {}. y_pred.shape {}'.format(y_true.shape, y_pred.shape)

    H, W, K = y_true.shape
    class_num = K

    TPs = np.zeros(class_num, np.int)
    FPs = np.zeros(class_num, np.int)
    FNs = np.zeros(class_num, np.int)
    TNs = np.zeros(class_num, np.int)
    ACCs = np.zeros(class_num)
    Ps = np.zeros(class_num)
    Rs = np.zeros(class_num)
    F1s = np.zeros(class_num)
    IoUs = np.zeros(class_num)
    for i in range(class_num):
        tp, fp, fn, tn, accuracy, precision, recall, f1, iou = segment_evaluate_binary(y_true[:, :, i], y_pred[:, :, i])
        TPs[i] = tp
        FPs[i] = fp
        FNs[i] = fn
        TNs[i] = tn
        ACCs[i] = accuracy
        Ps[i] = precision
        Rs[i] = recall
        F1s[i] = f1
        IoUs[i] = iou
    return TPs, FPs, FNs, TNs, ACCs, Ps, Rs, F1s, IoUs


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

