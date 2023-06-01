# -*- encoding: utf-8 -*-
'''
@Time       : 12/28/21 10:35 AM
@Author     : Jiang.xx
@Email      : cxyth@live.com
@Description:
'''
import sys
sys.path.append('../')
import os
import cv2
import shutil
import torch
from tqdm import tqdm
from model import create_model
from torchsummary import summary

if __name__=="__main__":
    ckpt_files = ['../runs/Upp_efb6.val9/ckpt/checkpoint-epoch39.pth',
                  '../runs/Upp_efb6.val9/ckpt/checkpoint-epoch40.pth',
                  '../runs/Upp_efb6.val9/ckpt/checkpoint-epoch41.pth',
                  '../runs/Upp_efb6.val9/ckpt/checkpoint-epoch42.pth',
                  '../runs/Upp_efb6.val9/ckpt/checkpoint-epoch43.pth',
                  '../runs/Upp_efb6.val9/ckpt/checkpoint-epoch44.pth']
    n_class = 10
    model = create_model('efficientnet-b6', n_class).cuda()
    model = torch.nn.DataParallel(model)
    ckpts = [torch.load(f)['state_dict'] for f in ckpt_files]
    avg_dict = {}
    for key in tqdm(ckpts[0].keys()):
        if key.endswith('.num_batches_tracked'):
            avg_dict[key] = ckpts[0][key]
        else:
            tensors = torch.cat([ckpt[key].unsqueeze(0) for ckpt in ckpts], axis=0)
            avg_dict[key] = tensors.mean(axis=0)
    model.load_state_dict(avg_dict)
    summary(model, input_size=(6, 256, 256))

    # model = create_model('efficientnet-b6', n_class).cuda()
    # model = torch.nn.DataParallel(model)
    # ckpt = torch.load(ckpt_files[-1])
    # model.load_state_dict(ckpt['state_dict'])
    # summary(model, input_size=(6, 256, 256))

