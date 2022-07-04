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
import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dataset import get_val_transform
from models import create_model
from utils import load_config, randering_mask
from infer import pred_large_imagery_by_sliding_window

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # segment root directory


class Segmenter(object):

    def __init__(self, cfg_name):
        cfg = load_config(cfg_name, 'configs')
        self.net_params = cfg['network_params']
        self.infer_params = cfg['inference_params']
        self.class_info = cfg['dataset_params']['cls_info']
        self.n_class = len(self.class_info.items())
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        nn_type = self.net_params['type']
        arch = self.net_params['arch']
        encoder = self.net_params['encoder']
        in_height = self.net_params['in_height']
        in_width = self.net_params['in_width']
        in_channel = self.net_params['in_channel']
        out_channel = self.net_params['out_channel']
        assert self.n_class == out_channel
        assert in_width == in_height
        self.input_size = in_width
        ckpt = ROOT / cfg['run_dir'] / cfg_name / "ckpt" / cfg['inference_params']['ckpt_name']
        print('\n[segmenter] load weights from {}\n'.format(ckpt))
        self.model = create_model(
            type=nn_type,
            arch=arch,
            encoder=encoder,
            in_channel=in_channel,
            out_channel=out_channel,
        ).to(self.device)
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        # summary(model, input_size=(in_channel, in_height, in_width))
        self.transform = get_val_transform(in_width)

    @torch.no_grad()
    def predict(self, rgb_img):
        assert rgb_img.shape[0] == self.input_size and rgb_img.shape[1] == self.input_size
        input = self.transform(image=rgb_img)['image']
        input = input.unsqueeze(0)
        input = input.to(self.device)
        output = self.model(input)    # (n, c, h, w)
        output = torch.softmax(output, dim=1)
        output = output.squeeze(0).cpu().numpy()
        pred = np.argmax(output, axis=0).astype(np.uint8)  # (h, w)
        return pred

    def predict_folder(self, in_dir, out_dir):
        # if not os.path.exists(out_dir): os.makedirs(out_dir)
        files = [f for f in os.listdir(in_dir) if f.endswith('.tif')]
        tbar = tqdm(files)
        tbar.set_description('Inference')
        for fname in tbar:
            image = cv2.imread(os.path.join(in_dir, fname))
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pred_mask = self.predict(rgb_img)

            out_path = os.path.join(out_dir, fname[:-4] + '.png')
            cv2.imwrite(out_path, pred_mask)

            if True:
                colors_bgr = [[0, 0, 0], [255, 0, 0]]
                res = randering_mask(image, pred_mask, self.n_class, colors_bgr, alpha=0.7, beta=0.3)
                res_name = out_path.replace('.png', '_cover.jpg')
                cv2.imwrite(res_name, res)

    def predict_large_imagery(self, in_path, out_path):
        pred_large_imagery_by_sliding_window(
            self.model,
            in_path=in_path,
            out_path=out_path,
            l1_win_sz=self.infer_params['l1_win_sz'],
            l1_overlap=self.infer_params['l1_overlap'],
            l2_win_sz=self.infer_params['l2_win_sz'],
            l2_overlap=self.infer_params['l2_overlap'],
            batch_sz=self.infer_params['batch_size'],
            transform=self.transform,
            n_class=self.n_class
        )

    def predict_large_imagery_folder(self, in_dir, out_dir):
        files = [f for f in os.listdir(in_dir) if f.endswith('.tif')]
        tbar = tqdm(files)
        tbar.set_description('Inference Large Imagery')
        for fname in tbar:
            img_path = os.path.join(in_dir, fname)
            out_path = os.path.join(out_dir, fname[:-4] + '.tif')
            self.predict_large_imagery(img_path, out_path)


if __name__ == '__main__':
    cfg = 'upp_efb5.v2'
    segmenter = Segmenter(cfg)

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    assert os.path.exists(in_dir) and os.path.exists(out_dir)
    segmenter.predict_folder(in_dir, out_dir)


