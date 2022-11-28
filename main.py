# -*- encoding: utf-8 -*-
'''
@Time       : 12/28/21 10:35 AM
@Author     : Jiang.xx
@Email      : cxyth@live.com
@Description: 通过main.py调用train.py和infer.py
'''
import os
import shutil
import torch
import argparse
from utils import load_config, save_config


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        type=str,
        help='Name of the config file, excluding the .json file extension.')
    argparser.add_argument(
        '-m', '--mode',
        default="train",
        type=str,
        choices=['train', 'infer'],
        help='Mode to launch the script in. '
             'Train: train model. '
             'Infer: inference on specified fold')
    argparser.add_argument('-g', '--gpus', default='0', type=str, help='gpus')

    # argparser.add_argument('--master_addr', default="localhost", type=str, help="Address of master node")
    # argparser.add_argument('--master_port', default="6666", type=str, help="Port on master node")
    # argparser.add_argument('-n', '--nodes', default=1, type=int, metavar='N', help="Number of total nodes")
    # argparser.add_argument('-g', '--gpus', default=1, type=int, help='Number of gpus per node')
    # argparser.add_argument('-nr', '--nr', default=0, type=int, help='Ranking within the nodes')

    args = argparser.parse_args()
    return args


def main():

    # --- Process args --- #
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # --- Process config --- #
    cfg = load_config(args.config, "configs")
    cfg['mode'] = args.mode
    cfg['run_name'] = args.config
    for item in cfg.items(): print(item)

    # setup log dir and save config
    log_dir = os.path.join(cfg['run_dir'], cfg['run_name'])
    if cfg['mode'] == "train":
        if os.path.exists(log_dir):
            while(True):
                _input = input(f"> Folder [{log_dir}] already exists, overwrite? (Y/N):")
                if _input in ['y', 'Y']:
                    shutil.rmtree(log_dir)
                    break
                elif _input in ['n', 'N']:
                    exit(0)
        os.makedirs(log_dir)
        save_config(cfg, log_dir)

    torch.manual_seed(0)
    if cfg['mode'] == "train":
        from train import train
        train(cfg)
    elif cfg['mode'] == "infer":
        from infer import inference
        inference(cfg)


if __name__ == "__main__":
    main()
