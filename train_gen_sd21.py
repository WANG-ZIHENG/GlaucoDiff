import argparse
import gc
import os

import numpy as np
import pandas as pd
import torch

from cldm.model import create_model
from cycle_train import gen_new_image, get_last_global, train_control
from share import *


def build_parser():
    parser = argparse.ArgumentParser(description='GlaucoDiff — train ControlNet on SLO fundus')
    parser.add_argument('--dataset_dir', default='./data/10k', type=str,
                        help='dataset root (contains data_summary.csv, filter_file.txt, All/)')
    parser.add_argument('--result_dir', default='./output/glaucodiff', type=str)
    parser.add_argument('--use_filter', default=True, type=bool)
    parser.add_argument('--seed', default=None, type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--train_samples_per_epoch', default=3, type=int)
    parser.add_argument('--gen_ddim_steps', default=25, type=int)
    return parser


def collect_training_files(args):
    summary_path = os.path.join(args.dataset_dir, 'data_summary.csv')
    summary_data = pd.read_csv(summary_path)
    train_list = summary_data[summary_data['use'] == 'training']['filename'].values
    return [os.path.join(args.dataset_dir, 'All', i) for i in train_list]


def main():
    args = build_parser().parse_args()

    if args.seed is None:
        args.seed = int(np.random.randint(100000, size=1)[0])
    args.result_dir = os.path.join(args.result_dir, f'seed{args.seed}')
    os.makedirs(args.result_dir, exist_ok=True)

    if os.path.exists(os.path.join(args.result_dir, 'log_train.txt')):
        global_epoch = get_last_global(args.result_dir) + 1
    else:
        global_epoch = 0

    data = collect_training_files(args)
    created_model = create_model('./models/cldm_v21.yaml').cpu()

    for _ in range(args.epochs):
        li = list(np.random.choice(data,
                                   size=min(args.train_samples_per_epoch, len(data)),
                                   replace=False))
        train_control(args=args, global_epoch=global_epoch,
                      topn_file=li, created_model=created_model)
        torch.cuda.empty_cache()
        gc.collect()

        gen_new_image(args=args, topn_file=li, created_model=created_model,
                      global_epoch=global_epoch, ddim_steps=args.gen_ddim_steps)
        torch.cuda.empty_cache()
        gc.collect()
        global_epoch += 1


if __name__ == '__main__':
    main()
