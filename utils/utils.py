# -*- encoding: utf-8 -*-
'''
@Time       : 12/28/21 10:35 AM
@Author     : Jiang.xx
@Email      : cxyth@live.com
@Description:
'''
import os
import time
import torch
import yaml
import logging


def load_config(config_name="config", config_dir=""):
    if os.path.splitext(config_name)[1] == ".yml":
        filepath = os.path.join(config_dir, config_name)
    else:
        filepath = os.path.join(config_dir, config_name + ".yml")

    if not os.path.isfile(filepath):
        raise FileNotFoundError('config file {} was not found.'.format(filepath))
    with open(filepath, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, config_dir):
    filepath = os.path.join(config_dir, 'config.yml')
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ExpSmoothing:
    def __init__(self, w=0.9):
        self.last = None
        self.w = w

    def reset(self):
        self.last = None

    def __call__(self, v):
        if self.last is None:
            self.last = v
        smoothed_val = self.last * self.w + (1 - self.w) * v
        self.last = smoothed_val
        return smoothed_val


def exp_smoothing(v, w=0.9):
    last = v[0]
    smoothed = []
    for point in v:
        smoothed_val = last * w + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def second2time(second):
    if second < 60:
        return str('{}'.format(round(second, 4)))
    elif second < 60*60:
        m = second//60
        s = second % 60
        return str('{}:{}'.format(int(m), round(s, 1)))
    elif second < 60*60*60:
        h = second//(60*60)
        m = second % (60*60)//60
        s = second % (60*60) % 60
        return str('{}:{}:{}'.format(int(h), int(m), int(s)))


def swa(ckpt_files):
    ckpts = [torch.load(f)['state_dict'] for f in ckpt_files]
    avg_dict = {}
    for key in ckpts[0].keys():
        if key.endswith('.num_batches_tracked'):
            avg_dict[key] = ckpts[0][key]
        else:
            tensors = torch.cat([ckpt[key].unsqueeze(0) for ckpt in ckpts], axis=0)
            avg_dict[key] = tensors.mean(axis=0)
    return avg_dict


def init_logger(file):
    logger = logging.getLogger('log')
    logger.setLevel(level=logging.INFO)
    basic_fmt = "%(asctime)s: %(message)s"
    date_fmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt=basic_fmt, datefmt=date_fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(file)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def print_excel(excelData, filename):
    '''
        打印到excel表格
    :param excelData: 需要输出到表格的格式化数据
    :param filename: excel表格的文件名
    :return:
    '''
    import xlwt
    # 创建工作簿
    workbook = xlwt.Workbook(encoding='utf-8')
    # 创建sheet
    data_sheet = workbook.add_sheet('demo', cell_overwrite_ok=True)

    # 创建表格样式
    file_style = xlwt.XFStyle()

    # 定义输出的行数
    index = 0
    for i in range(len(excelData)):
        # 每一列的内容(i)
        for j in range(len(excelData[i])):
            data_sheet.write(index, 0, i, file_style)
            data_sheet.write(index, 1, excelData[i][j][0], file_style)
            data_sheet.write(index, 2, excelData[i][j][1], file_style)
            index += 1

    # 保存文件
    filename = filename + ".xls"
    workbook.save(filename)
    print("打印成功")


if __name__ == '__main__':
    logger = init_logger('test.log')
    for i in range(6):
        logger.info('{} hello!'.format(i))

