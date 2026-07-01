import argparse
import os
import re
from glob import glob

from cldm.model import create_model
from pipeline import gen_new_image
from share import *


def build_parser():
    parser = argparse.ArgumentParser(description='GlaucoDiff — generate vCDR-controlled SLO fundus images')
    parser.add_argument('--dataset_dir', default='./data/fairvlmed10k', type=str,
                        help='dataset root containing All/*.npz and mask/*_predict.png')
    parser.add_argument('--ckpt_path', required=True, type=str,
                        help='path to a trained ControlNet checkpoint (.ckpt)')
    parser.add_argument('--result_dir', default='./output/generated', type=str)
    parser.add_argument('--use_filter', default=True, type=bool)
    parser.add_argument('--ddim_steps', nargs='+', default=[150, 200], type=int)
    parser.add_argument('--files', nargs='*', default=None,
                        help='optional explicit list of .npz files to condition on; '
                             'if omitted, all files under dataset_dir/All are used')
    return parser


def main():
    args = build_parser().parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    if args.files:
        topn_file = args.files
    else:
        topn_file = glob(os.path.join(args.dataset_dir, 'All', '*.npz'))
        if not topn_file:
            topn_file = glob(os.path.join(args.dataset_dir, '*', '*.npz'))

    m = re.findall(r'epoch=(\d+)', args.ckpt_path)
    global_epoch = int(m[0]) if m else 0

    created_model = create_model('./models/cldm_v21.yaml').cpu()

    for steps in args.ddim_steps:
        gen_new_image(args=args, topn_file=topn_file,
                      created_model=created_model,
                      global_epoch=global_epoch,
                      ckpt_path=args.ckpt_path,
                      ddim_steps=steps,
                      save_dir=os.path.join(args.result_dir, f'sd_gen_step_{steps}'))


if __name__ == '__main__':
    main()
