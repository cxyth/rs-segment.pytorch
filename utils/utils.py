# -*- encoding: utf-8 -*-
'''
@Time       : 12/28/21 10:35 AM
@Author     : Jiang.xx
@Email      : cxyth@live.com
@Description: 一些实用工具
'''
import os
import time
import logging


class AverageMeter(object):
    def __init__(self):
        self.reset()

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


def exp_smoothing(v, w=0.9):
    last = v[0]
    smoothed = []
    for point in v:
        smoothed_val = last * w + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


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
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(file)
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

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

