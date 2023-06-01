# -*- encoding: utf-8 -*-
'''
@Time       : 07/04/22 10:35 AM
@Author     : Jiang.xx
@Email      : cxyth@live.com
@Description: 推理
'''
import os
import sys
import cv2
import torch
import argparse
import ttach as tta
import numpy as np
import os.path as osp
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.ImgMaskDataset import get_val_transform
from datasets.ImgDataset import ImageDataset
from models import create_model
from utils import load_config, sementic_splash
from utils.pred import RSImagePredictManager, WeightedPredictManager

# to solve the problem of 'ERROR 1: PROJ: pj_obj_create: Open of /opt/conda/share/proj failed'
# os.environ['PROJ_LIB'] = '/opt/conda/share/proj'
os.environ['PROJ_LIB'] = r'C:\Users\AI\anaconda3\envs\torch17\Library\share\proj'


class Segmenter(object):

    def __init__(self, cfg_name, weight):
        cfg = load_config(cfg_name, 'configs')
        self.cfgN = cfg['network']
        self.cfgN['pretrained'] = None
        self.cfgD = cfg['dataset']
        self.cfgI = cfg['infer']
        self.n_class = len(self.cfgD['cls_info'])
        self.transform = get_val_transform()
        log_dir = osp.join(cfg['run_dir'], cfg_name)
        #  check weight file
        ckpt_path = osp.join(log_dir, 'ckpt', weight)
        if not osp.exists(ckpt_path):
            print(f'file {ckpt_path} not found.')
            return
        self.model = create_model(cfg=self.cfgN).cuda()
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        # summary(model, input_size=(in_channel, in_height, in_width))
        if self.cfgI['tta']:
            tta_transforms = tta.aliases.d4_transform()
            # tta_transforms = tta.Compose(
            #     [
            #         tta.HorizontalFlip(),
            #         tta.VerticalFlip(),
            #         tta.Rotate90(angles=[0, 180]),
            #         tta.Scale(scales=[1, 1.5, 2]),
            #     ]
            # )
            self.model = tta.SegmentationTTAWrapper(self.model, tta_transforms, merge_mode='mean')

    @torch.no_grad()
    def predict(self, rgb_img):
        input = self.transform(image=rgb_img)['image']
        input = input.unsqueeze(0).cuda()
        output = self.model(input)    # (n, c, h, w)
        output = torch.softmax(output, dim=1)
        output = output.squeeze(0).cpu().numpy()
        output = np.argmax(output, axis=0).astype(np.uint8)  # (h, w)
        return output

    def predict_folder(self, in_dir, out_dir):
        # per image
        os.makedirs(out_dir, exist_ok=True)
        files = [f for f in os.listdir(in_dir) if f.endswith('.tif')]
        tbar = tqdm(files)
        tbar.set_description('Inference')
        for fname in tbar:
            image = cv2.imread(os.path.join(in_dir, fname))
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pred_mask = self.predict(rgb_img)
            out_path = os.path.join(out_dir, fname[:-4] + '.png')
            cv2.imwrite(out_path, pred_mask.astype(np.uint8))

            # colors_bgr = [[0, 0, 0], [255, 0, 0]]
            res = sementic_splash(image, pred_mask, n_label=self.n_class, alpha=1.0, beta=0.5)
            res_name = out_path.replace('.png', '_color.jpg')
            cv2.imwrite(res_name, res)

    def predict_large_imagery(self, in_path, out_path):

        rs_mng = RSImagePredictManager(
            in_raster=in_path,
            out_raster=out_path,
            window_sz=self.cfgI['l1_win_sz'],
            net_sz=self.cfgI['l2_win_sz'],
            overlap=self.cfgI['l1_overlap'])

        tbar = tqdm(range(len(rs_mng)))
        tbar.set_description(os.path.basename(in_path))
        for _ in tbar:
            imdata_chw, _ = rs_mng.get_next()
            im_loader = DataLoader(
                dataset=ImageDataset(
                    im_data=imdata_chw,
                    tile_size=self.cfgI['l2_win_sz'],
                    overlap=self.cfgI['l2_overlap'],
                    transform=self.transform,
                    channel_first=True),
                batch_size=self.cfgI['batch_size'],
                shuffle=False,
                num_workers=2,
                drop_last=False)
            pred_mng = WeightedPredictManager(
                map_height=imdata_chw.shape[1],
                map_width=imdata_chw.shape[2],
                map_channel=self.n_class,
                patch_height=self.cfgI['l2_win_sz'],
                patch_width=self.cfgI['l2_win_sz'])
            with torch.no_grad():
                for batch_data in im_loader:
                    patch, y1y2x1x2 = batch_data['image'], batch_data['window']
                    patch = patch.cuda()
                    output = self.model(patch)
                    # output = F.interpolate(output, size=(patch_size, patch_size), mode='bilinear')  # -> (n, c, h, w)
                    probs = torch.softmax(output, dim=1).cpu().numpy()
                    pred_mng.update(probs, y1y2x1x2.numpy())

            pred_mask, _ = pred_mng.get_result()
            # post processing

            rs_mng.update(pred_mask[None, :, :])
        rs_mng.close()

    def predict_large_imagery_folder(self, in_dir, out_dir):
        files = [f for f in os.listdir(in_dir) if f.endswith('.tif')]
        tbar = tqdm(files)
        tbar.set_description('Inference Large Imagery')
        for fname in tbar:
            img_path = os.path.join(in_dir, fname)
            out_path = os.path.join(out_dir, fname[:-4] + '.tif')
            self.predict_large_imagery(img_path, out_path)


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-c', '--config', type=str, required=True, help='Name of the config file.')
    argparser.add_argument('-w', '--weight', default='best.pt', type=str, help='Choice a weight file.')
    argparser.add_argument('-i', '--input', required=True, type=str)
    argparser.add_argument('-o', '--output', required=True, type=str)
    argparser.add_argument('-g', '--gpus', default='0', type=str, help='gpus')
    return argparser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    segmenter = Segmenter(args.config, args.weight)
    # segmenter.predict_folder(args.input, args.output)
    segmenter.predict_large_imagery_folder(args.input, args.output)


