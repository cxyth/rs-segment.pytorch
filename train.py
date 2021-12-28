# -*- encoding: utf-8 -*-
'''
@Time       : 12/28/21 10:35 AM
@Author     : Jiang.xx
@Email      : cxyth@live.com
@Description: 训练
'''
import os
import time
import copy
import torch
import random
import numpy as np
import os.path as osp
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler # need pytorch>1.6
import matplotlib.pyplot as plt
from pytorch_toolbelt import losses as L
from utils.utils import AverageMeter, init_logger, exp_moothing
from utils.metric import Metric
from utils.cutmix import CutMixCollator
from dataset import myDataset, get_sample_weights, get_train_transform, get_val_transform
from models import create_model
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, SoftCrossEntropyLoss, SoftBCEWithLogitsLoss


def train(CFG):
    # 参数
    D = CFG['dataset_params']
    class_info      = D['cls_info']
    n_class         = len(class_info.items())
    train_dirs      = D['train_dirs']
    val_dirs        = D['val_dirs']
    img_ext         = D['image_ext']
    resample        = D['resample']

    T = CFG['train_params']
    epochs          = T['epochs']
    batch_size      = T['batch_size']
    lr              = T['lr']
    smoothing       = T['smoothing']
    weight_decay    = T['weight_decay']
    save_inter      = T['save_inter']
    min_inter       = T['min_inter']
    iter_inter      = T['iter_inter']
    plot            = T['plot']
    log_dir         = os.path.join(CFG['run_dir'], CFG['run_name'])
    ckpt_dir        = os.path.join(log_dir, 'ckpt')

    N = CFG['network_params']
    nn_type         = N['type']
    arch            = N['arch']
    encoder         = N['encoder']
    in_height       = N['in_height']
    in_width        = N['in_width']
    in_channel      = N['in_channel']
    out_channel     = N['out_channel']
    pretrained      = N['pretrained']
    assert n_class >= 2
    assert n_class == out_channel
    assert in_width == in_height

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 准备数据集
    train_transform = get_train_transform(in_width)
    val_transform = get_val_transform(in_width)
    train_data = myDataset(train_dirs, train_transform, img_ext)
    valid_data = myDataset(val_dirs, val_transform, img_ext)
    train_data_size = train_data.__len__()
    valid_data_size = valid_data.__len__()
    img_c, img_h, img_w = train_data.__getitem__(0)['image'].shape
    assert img_c == in_channel, f'in_channel:{in_channel} img_c:{img_c}'


    if resample:
        sample_weights = get_sample_weights(train_dirs, n_class)
        sampler = WeightedRandomSampler(sample_weights, num_samples=sample_weights.size, replacement=True)
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=16,
            sampler=sampler)
    else:
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=batch_size,
            drop_last=True)
    valid_loader = DataLoader(
        dataset=valid_data, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=batch_size,
        drop_last=True)

    # logger
    logger = init_logger(os.path.join(log_dir, time.strftime("%m-%d-%H-%M-%S", time.localtime()) + '.log'))
    logger.info('Total Epoch:{} Image_size:({}, {}) Training num:{}  Validation num:{}'
                .format(epochs, img_w, img_h, train_data_size, valid_data_size))

    # 网络
    model = create_model(type=nn_type,
                         arch=arch,
                         encoder=encoder,
                         in_channel=in_channel,
                         out_channel=out_channel,
                         pretrained=pretrained).cuda()
    model = torch.nn.DataParallel(model)

    # 训练
    scaler = GradScaler()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5, last_epoch=-1)

    DiceLoss_fn = DiceLoss(mode='multiclass')
    SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=smoothing)
    criterion = L.JointLoss(first=DiceLoss_fn, second=SoftCrossEntropy_fn,
                              first_weight=0.5, second_weight=0.5).cuda()

    train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = [], [], []
    train_loader_size = train_loader.__len__()
    valid_loader_size = valid_loader.__len__()
    best_iou = 0
    best_mode = copy.deepcopy(model)
    epoch_start = 0

    # 主循环
    for epoch in range(epoch_start, epochs):
        t0 = time.time()
        # 训练阶段
        model.train()
        train_epoch_loss = AverageMeter()
        train_iter_loss = AverageMeter()
        for batch_idx, batch_samples in enumerate(train_loader):
            data, target = batch_samples['image'], batch_samples['label']
            data, target = data.cuda(), target.cuda()
            # --------------------------------
            # with autocast():    # need pytorch > 1.6
            pred = model(data)
            loss = criterion(pred, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # --------------------------------
            scheduler.step(epoch + batch_idx / train_loader_size) 
            image_loss = loss.item()
            train_epoch_loss.update(image_loss)
            train_iter_loss.update(image_loss)
            if batch_idx % iter_inter == 0:
                spend_time = time.time() - t0
                logger.info('[train] epoch:{} iter:{}/{} {:.2f}% lr:{:.6f} loss:{:.6f} ETA:{}min'.format(
                    epoch, batch_idx, train_loader_size, batch_idx/train_loader_size*100,
                    optimizer.param_groups[-1]['lr'],
                    train_iter_loss.avg, spend_time / (batch_idx+1) * train_loader_size // 60 - spend_time // 60))
                train_iter_loss.reset()

        # 验证阶段
        model.eval()
        valid_epoch_loss = AverageMeter()
        valid_iter_loss = AverageMeter()
        M = Metric(num_class=n_class)
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(valid_loader):
                data, target = batch_samples['image'], batch_samples['label']
                data, target = data.cuda(), target.cuda()
                # --------------------------------
                # with autocast():  # need pytorch > 1.6
                pred = model(data)
                loss = criterion(pred, target)
                # --------------------------------
                pred = torch.softmax(pred, dim=1).cpu().numpy()
                pred = np.argmax(pred, axis=1)
                M.add_batch(pred, target.cpu().numpy())

                image_loss = loss.item()
                valid_epoch_loss.update(image_loss)
                valid_iter_loss.update(image_loss)

            val_loss = valid_iter_loss.avg
            scores = M.evaluate()
            # Ps = scores['class_precision']
            # Rs = scores['class_recall']
            # IoUs = scores['class_iou']
            mIoU = scores['mean_iou']
            logger.info('[val] epoch:{} loss:{:.6f} miou:{:.2f}'.format(epoch, val_loss, mIoU))

        # 保存loss、lr
        train_loss_total_epochs.append(train_epoch_loss.avg)
        valid_loss_total_epochs.append(valid_epoch_loss.avg)
        epoch_lr.append(optimizer.param_groups[0]['lr'])
        # 保存模型
        if epoch % save_inter == 0 and epoch > min_inter:
            state = {'epoch': epoch, 'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(ckpt_dir, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)  # pytorch1.6会压缩模型，低版本无法加载
        # 保存最优模型
        if mIoU > best_iou:  # train_loss_per_epoch valid_loss_per_epoch
            state = {'epoch': epoch, 'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(ckpt_dir, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_iou = mIoU
            best_mode = copy.deepcopy(model)
            logger.info('[save] Best Model saved at epoch:{}'.format(epoch))
        logger.info('==============================================================')

    # 训练loss曲线
    if plot:
        x = [i for i in range(epochs)]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, exp_moothing(train_loss_total_epochs, 0.6), label='train loss')
        ax.plot(x, exp_moothing(valid_loss_total_epochs, 0.6), label='val loss')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('CrossEntropy', fontsize=15)
        ax.set_title('train curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, epoch_lr,  label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Learning Rate', fontsize=15)
        ax.set_title('lr curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.savefig(osp.join(log_dir, 'plot.png'))
        plt.show()
            
    return best_mode, model

