import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator, Dataset10k, WrapperDataset, Datasetfairvlmed10k, \
    dataset_sd_gen
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from glob import glob
from torchvision import transforms
import pandas as pd
parser = argparse.ArgumentParser()

if os.path.exists("/D_share"):
    parser.add_argument('--root_path', type=str,
                        default='/H_share/data/gen_image', help='root dir for data')
else:
    parser.add_argument('--root_path', default='/root/autodl-tmp/gen_image/10k', type=str)


parser.add_argument('--dataset', type=str,
                    default='sd_gen0', help='10k or fairvlmed10k or sd_gen0')
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    root_path = os.path.join(args.root_path, args.dataset)
    test_list = []
    if args.dirs == None:
        test_root_dir = root_path
        if args.dataset == "sd_gen0":
            test_list.extend(glob(os.path.join(test_root_dir, "*generate.png")))
            test_list = [i for i in test_list if "extra" in i]

    else:
        for dir in args.dirs:
            test_root_dir = os.path.join(root_path, dir)
            if args.dataset == "sd_gen0":
                test_list.extend(glob(os.path.join(test_root_dir, "*generate.png")))
                test_list = [i for i in test_list if "extra" in i]
            else:
                test_list.extend(glob(os.path.join(test_root_dir, "*.npz")))


    db_test = args.Dataset(args,test_list)

    db_test = args.Wrapper(data_class=db_test,
                              transform=transforms.Compose(
                                   [RandomGenerator(output_size=None,mode="test")]),dataset = args.dataset)

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=8)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    pandas_list = []
    for i_batch, sampled_batch in enumerate(tqdm(testloader)):
        # h, w = sampled_batch["image"].size()[1:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model,loader=testloader, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        pandas_list.append([case_name,np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1],np.mean(metric_i, axis=0)[2]])
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f mean_mse %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1],np.mean(metric_i, axis=0)[2]))

    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f mean_mse %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_mse = np.mean(metric_list, axis=0)[2]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f mean_mse : %f' % (performance, mean_hd95,mean_mse))
    seed = int(np.random.randint(100000, size=1)[0])
    df = pd.DataFrame(pandas_list,columns=['case_name','dice','hd95','mse'])
    df.to_csv(os.path.join(root_path,"../",f"sd_gen_metric{seed}.csv"),index=False)
    return "Testing Finished!"


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        '10k': {
            'Dataset': Dataset10k,
            'Wrapper':WrapperDataset,
            'volume_path': args.root_path,
            'num_classes': 3,
            'z_spacing': 1,
        },
        'sd_gen0': {
            'Dataset': dataset_sd_gen,
            'Wrapper': WrapperDataset,
            'volume_path': args.root_path,
            'num_classes': 3,
            'z_spacing': 1,
            'only_gen': True,
            'dirs': None
        },
        'fairvlmed10k': {
            'Dataset': Datasetfairvlmed10k,
            'Wrapper':WrapperDataset,
            'volume_path': args.root_path,
            'num_classes': 3,
            'z_spacing': 1,
            'only_gen':True,
            'dirs' : ['Training','Validation',"Test"]
        },
    }
    dataset_name = args.dataset
    args.Wrapper = dataset_config[dataset_name]['Wrapper']
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.only_gen = dataset_config[dataset_name]['only_gen']
    args.dirs = dataset_config[dataset_name]['dirs']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + "10k" + str(args.img_size)
    snapshot_path = "model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    # snapshot = "model/TU_10k224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth"

    net.load_state_dict(torch.load(snapshot))
    # net.load_from(weights=np.load(config_vit.pretrained_path))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = 'predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)


