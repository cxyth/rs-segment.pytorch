# -*- encoding: utf-8 -*-
'''
@Time       : 12/28/21 10:35 AM
@Author     : Jiang.xx
@Email      : cxyth@live.com
@Description: 推理
'''
import os
import torch
from tqdm import tqdm
from torchsummary import summary
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.pred import RSImageSlideWindowManager, ImageDataset, WeightedPredictManager
from dataset import get_val_transform
from models import create_model

# to solve the problem of 'ERROR 1: PROJ: pj_obj_create: Open of /opt/conda/share/proj failed'
# os.environ['PROJ_LIB'] = '/opt/conda/share/proj'
os.environ['PROJ_LIB'] = r'C:\Users\AI\anaconda3\envs\torch17\Library\share\proj'


def pred_large_imagery_by_sliding_window(
        model,
        in_path,
        out_path,
        l1_win_sz,
        l1_overlap,
        l2_win_sz,
        l2_overlap,
        batch_sz,
        transform,
        n_class,
        TTA=False
):

    assert TTA == False, 'TTA not implement'
    rs_mng = RSImageSlideWindowManager(
        in_raster=in_path,
        out_raster=out_path,
        window_sz=l1_win_sz,
        net_sz=l2_win_sz,
        overlap=l1_overlap)

    tbar = tqdm(range(len(rs_mng)))
    tbar.set_description(os.path.basename(in_path))
    for _ in tbar:
        imdata_chw, _ = rs_mng.get_next()
        im_loader = DataLoader(
            dataset=ImageDataset(
                im_data=imdata_chw,
                tile_size=l2_win_sz,
                overlap=l2_overlap,
                transform=transform,
                channel_first=True),
            batch_size=batch_sz,
            shuffle=False,
            num_workers=batch_sz,
            drop_last=False)
        pred_mng = WeightedPredictManager(
            map_height=imdata_chw.shape[1],
            map_width=imdata_chw.shape[2],
            map_channel=n_class,
            patch_height=l2_win_sz,
            patch_width=l2_win_sz)
        with torch.no_grad():
            for batch_data in im_loader:
                patch, windows = batch_data['image'], batch_data['window']
                patch = patch.cuda()
                output = model(patch)
                # output = F.interpolate(output, size=(patch_size, patch_size), mode='bilinear')  # -> (n, c, h, w)
                probs = torch.softmax(output, dim=1).cpu().numpy()
                pred_mng.update(probs, windows.numpy())

        pred_mask, _ = pred_mng.get_result()

        # post processing
        # assert n_class == 2
        # pred_mask = mask_cleanout_small_chip(pred_mask, 32*32)
        # pred_mask = ~pred_mask.astype(np.bool)
        # pred_mask = mask_cleanout_small_chip(pred_mask, 32*32)
        # pred_mask = ~pred_mask.astype(np.bool)
        # pred_mask = pred_mask.astype(np.uint8)

        rs_mng.fit_result(pred_mask[None, :, :])
    rs_mng.close()


def inference(CFG):
    D = CFG['dataset_params']
    class_info      = D['cls_info']
    n_class         = len(class_info.items())
    assert n_class >= 2

    cfgN            = CFG['network_params']
    cfgN['pretrained'] = None

    I = CFG['inference_params']
    ckpt            = os.path.join(CFG['run_dir'], CFG['run_name'], "ckpt", I['ckpt_name'])
    input_dir       = I['in_dir']
    base_dir        = os.path.join(CFG['run_dir'], CFG['run_name'], I['out_dir'])
    res_dir         = os.path.join(base_dir, 'results')
    l1_win_sz       = I['l1_win_sz']
    l1_overlap      = I['l1_overlap']
    l2_win_sz       = I['l2_win_sz']
    l2_overlap      = I['l2_overlap']
    batch_size      = I['batch_size']
    TTA             = I['tta']

    os.makedirs(res_dir, exist_ok=True)

    # network
    model = create_model(cfg=cfgN).cuda()
    # model = torch.nn.DataParallel(model)
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    # summary(model, input_size=(in_channel, in_height, in_width))

    transform = get_val_transform(l2_win_sz)
    img_set = [f for f in os.listdir(input_dir) if f.endswith('.tif')]

    # Inference
    tbar = tqdm(img_set)
    tbar.set_description('Inference')
    for fid in tbar:
        # tbar.set_postfix_str(fid)
        img_path = os.path.join(input_dir, fid)
        out_path = os.path.join(res_dir, fid[:-4] + '.tif')
        pred_large_imagery_by_sliding_window(
            model,
            in_path=img_path,
            out_path=out_path,
            l1_win_sz=l1_win_sz,
            l1_overlap=l1_overlap,
            l2_win_sz=l2_win_sz,
            l2_overlap=l2_overlap,
            batch_sz=batch_size,
            transform=transform,
            n_class=n_class,
            TTA=TTA)
