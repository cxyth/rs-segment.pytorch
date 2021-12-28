# -*- encoding: utf-8 -*-
'''
@Time       : 12/28/21 10:35 AM
@Author     : Jiang.xx
@Email      : cxyth@live.com
@Description: 推理/精度评估
'''
import os
import time
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torchsummary import summary
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils import randering_mask, read_gdal, write_gdal, uint16_to_8
from utils import Metric, plot_confusion_matrix
from dataset import myDataset, batchtest_Dataset, get_val_transform
from utils.segment import PredictManager
from models import create_model


def predict_in_large_RSimagery(model, image, tile_size, overlap, batch_size, transform, n_class, TTA=False):

    def _predict(img):
        t0 = time.time()
        img_h, img_w, _ = img.shape
        predmng = PredictManager(img_h, img_w, n_class, tile_size, tile_size)

        test_data = batchtest_Dataset(img, tile_size, overlap, transform)
        test_loader = DataLoader(
            dataset=test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=batch_size,
            drop_last=False)
        for batch_idx, batch_data in enumerate(tqdm(test_loader)):
            tiles, boxes = batch_data['image'], batch_data['box']
            boxes = list(np.concatenate([t.numpy()[:, None] for t in boxes], axis=1))
            with torch.no_grad():
                tiles = tiles.cuda()
                maps = model(tiles)
                maps = F.interpolate(maps, size=(tile_size, tile_size), mode='bilinear') # -> (n, c, h, w)
                outputs = torch.softmax(maps, dim=1).cpu().numpy()

            for i in range(outputs.shape[0]):
                y1, y2, x1, x2 = boxes[i]
                predmng.update(outputs[i], yoff=y1, xoff=x1)

        return predmng.get_result(), time.time() - t0

    if TTA:
        img_h, img_w, _ = image.shape
        prob_maps = np.zeros((n_class, img_h, img_w), dtype=np.float32)
        total_time = 0

        _map, _time = _predict(image)
        prob_maps += _map
        total_time += _time

        img_fr = np.fliplr(image)
        _map, _time = _predict(img_fr)
        _map = np.flip(_map, axis=2)
        prob_maps += _map
        total_time += _time

        img_fv = np.flipud(image)
        _map, _time = _predict(img_fv)
        _map = np.flip(_map, axis=1)
        prob_maps += _map
        total_time += _time

        img_fr_fv = np.flipud(img_fr)
        _map, _time = _predict(img_fr_fv)
        _map = np.flip(_map, axis=1)
        _map = np.flip(_map, axis=2)
        prob_maps += _map
        total_time += _time

        prob_maps = prob_maps / 4
    else:
        prob_maps, total_time = _predict(image)

    return prob_maps, total_time


def predict_in_tile_image(model, image, transform, n_class, TTA=False):

    def _predict(img):
        t0 = time.time()
        img = transform(image=img)['image']
        img = img.unsqueeze(0)
        with torch.no_grad():
            img = img.cuda()
            output = model(img)
        output = output.permute(0, 2, 3, 1)  # -> (n, h, w, c)
        output = torch.softmax(output, dim=3)
        output = output.squeeze().cpu().numpy()
        return output, time.time() - t0

    if TTA:
        img_h, img_w, _ = image.shape
        prob_maps = np.zeros((img_h, img_w, n_class), dtype=np.float32)
        total_time = 0

        _map, _time = _predict(image)
        prob_maps += _map
        total_time += _time

        img_fr = np.fliplr(image)
        _map, _time = _predict(img_fr)
        _map = np.fliplr(_map)
        prob_maps += _map
        total_time += _time

        img_fv = np.flipud(image)
        _map, _time = _predict(img_fv)
        _map = np.flipud(_map)
        prob_maps += _map
        total_time += _time

        img_fr_fv = np.flipud(img_fr)
        _map, _time = _predict(img_fr_fv)
        _map = np.flipud(_map)
        _map = np.fliplr(_map)
        prob_maps += _map
        total_time += _time

        prob_maps = prob_maps / 4
    else:
        prob_maps, total_time = _predict(image)

    return prob_maps, total_time



def inference(CFG):
    D = CFG['dataset_params']
    class_info      = D['cls_info']
    n_class         = len(class_info.items())
    colors          = D['cls_color']
    img_ext         = D['image_ext']

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

    I = CFG['inference_params']
    ckpt            = os.path.join(CFG['run_dir'], CFG['run_name'], "ckpt", I['ckpt_name'])
    input_dir       = I['in_dir']
    base_dir        = os.path.join(CFG['run_dir'], CFG['run_name'], I['out_dir'])
    res_dir         = os.path.join(base_dir, 'results')
    tile_size       = I['tile_size']
    overlap         = I['overlap']
    batch_size      = I['batch_size']
    TTA             = I['tta']
    draw_mask       = I['draw']
    evaluate        = I['evaluate']

    os.makedirs(res_dir, exist_ok=True)

    # network
    model = create_model(type=nn_type,
                         arch=arch,
                         encoder=encoder,
                         in_channel=in_channel,
                         out_channel=out_channel,
                         pretrained=pretrained).cuda()
    # model = torch.nn.DataParallel(model)
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    # summary(model, input_size=(in_channel, in_height, in_width))

    transform = get_val_transform(in_width)
    img_set = [f for f in os.listdir(input_dir) if f.endswith(img_ext)]

    # Inference
    # tbar = tqdm(img_set)
    # tbar.set_description('Inference')
    # for fid in tbar:
    #     # tbar.set_postfix_str(fid)
    #     img_path = os.path.join(input_dir, fid)
    #     image, im_proj, im_geotrans = read_gdal(img_path)
    #     img_h, img_w, _ = image.shape
    #
    #     # print("image loaded:", image.shape)
    #
    #     if img_h > in_height or img_w > in_width:
    #         probmap, t = predict_in_large_RSimagery(
    #             model,
    #             image=image,
    #             tile_size=tile_size,
    #             overlap=overlap,
    #             batch_size=batch_size,
    #             transform=transform,
    #             n_class=n_class,
    #             TTA=TTA)
    #     else :
    #         probmap, t = predict_in_tile_image(
    #             model,
    #             image=image,
    #             transform=transform,
    #             n_class=n_class,
    #             TTA=TTA)
    #
    #     # post processing
    #     pred_mask = np.argmax(probmap, axis=2).astype(np.uint8)
    #
    #     out_path = os.path.join(res_dir, fid)
    #     write_gdal(pred_mask[:, :, None], out_path, im_proj, im_geotrans)
    #
    #     if draw_mask:
    #         bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #         res = randering_mask(bgr_img, pred_mask, n_class, colors, alpha=0.8, beta=0.5)
    #         res_name = out_path.rstrip(img_ext) + '_cover.jpg'
    #         cv2.imwrite(res_name, res)

    # Evaluate
    if not evaluate: return
    txt = []
    cls_names = list(class_info.keys())
    M = Metric(n_class)
    tbar = tqdm(img_set)
    tbar.set_description('Evaluate')
    for fid in tbar:
        # tbar.set_postfix_str(fid)
        pred_path = os.path.join(res_dir, fid)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        label_path = os.path.join(input_dir.replace('images', 'labels'), fid)
        if os.path.isfile(label_path):
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            # label = (label > 0).astype(np.uint8)
        else:
            raise FileNotFoundError(f'file [{label_path}] not found.')
        M.add_batch(pred, label)

    scores = M.evaluate()
    mIoU = scores["mean_iou"]
    IoUs = scores["class_iou"]
    Ps = scores["class_precision"]
    Rs = scores["class_recall"]
    confusion_matrix = M.get_confusion_matrix()
    _output = '> mIoU: {:.4f}'.format(mIoU)
    print(_output)
    txt.append(_output + '\n')
    for i in range(n_class):
        _output = '{:<20}| IoU: {:<10.4f} P: {:<.4f} R: {:<.4f}'.format(cls_names[i], IoUs[i], Ps[i], Rs[i])
        print(_output)
        txt.append(_output + '\n')

    # write txt
    with open(os.path.join(base_dir, 'score.txt'), 'w', newline='') as f:
        f.writelines(txt)

    # confusion matrix to csv
    mcm_csv = pd.DataFrame(confusion_matrix, index=cls_names, columns=cls_names)
    mcm_csv.to_csv(os.path.join(base_dir, 'confusion_matrix.csv'))


