import pickle
import re

import math
import numpy as np
from torchvision import transforms, datasets, models
import os
import mydatasets

from collections import defaultdict
import random
from torch.utils import data
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from sklearn.metrics import *
from fairlearn.metrics import *
from autoaug import ImageNetPolicy
import argparse

def get_args():
    parser=argparse.ArgumentParser(description='Train the model on images and target labels',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e','--epochs', metavar='E', type=int, default=100, nargs='?', help='Number of epochs', dest='max_epochs')
    parser.add_argument('-b','--batch-size', metavar='B', type=int, default=32, nargs='?', help='Batch size', dest='batch_size')
    parser.add_argument('-l','--learning-rate', metavar='LR', type=float, default=1e-5, nargs='?', help='Learning rate', dest='lr')
    if os.path.exists("/D_share"):
        parser.add_argument('--data_root', type=str, default="/H_share/data")
    else:
        parser.add_argument('--data_root', type=str, default="/root/autodl-tmp")

    parser.add_argument('--model_task', type=str, default="label_classification", help='label_classification')
    parser.add_argument('--dataset', type=str, default="10k", help='10k or fairvlmed10k')
    parser.add_argument('--warmup_epochs', default=5, type=int,
                        help='warmup epochs')
    parser.add_argument('--cos', default=True, type=bool,
                        help='lr decays by cosine scheduler. ')


    parser.add_argument('--model',type=str, default='efficientnet-b0',choices=("DenseNet121","efficientnet-b0","efficientnet-b3","efficientnet-b7",'resnet10','resnet18','resnet32', 'resnet34', 'resnet50', 'resnext50_32x4d'))
    parser.add_argument('--pretrain_model', default=False,action='store_true')
    parser.add_argument('--use_fake_data', default=False, action='store_true')
    parser.add_argument('--top_precentege', default=1,type=float,help="使用生成数据的百分比")
    parser.add_argument('--balance_data', default=False, action='store_true')
    parser.add_argument('--balance_attribute', type=str, default="gender", help='gender,race,ethnicity,language,maritalstatus,age,label')
    parser.add_argument('--device',type=str, default='cuda')
    parser.add_argument('--best_model_path',type=str, default='')






    return parser.parse_args()


def adjust_lr(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup_epochs:
        lr = lr / args.warmup_epochs * (epoch + 1)
    elif args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs + 1) / (args.max_epochs - args.warmup_epochs + 1)))

    # if lr < 0.001:
    #     lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




class DatasetFairvlmed10k():

    def __init__(self,args,root_path,path_list,gen_list):
        filter_path, summary_path = os.path.join(root_path, "filter_file.txt"),os.path.join(root_path, "data_summary.csv")

        self.data = path_list
        self.summary_data = pd.read_csv(summary_path)
        self.summary_data = self.summary_data.rename(columns={'glaucoma':'label'})
        # self.summary_data = self.summary_data.rename(columns={'age':'age_detail'})

        def categorize_age(age):
            if age >= 65:
                return 'elderly'
            else:
                return 'young'

        # 使用apply方法创建新列age
        self.summary_data['age'] = self.summary_data['age_detail'].apply(categorize_age)


        if filter_path:
            with open(filter_path,'r') as f:
                filter_list = f.read()
                filter_list = filter_list.split("\n")

            filter_list = [i.replace(".png",".npz") for i in filter_list]
            filter_list = set(filter_list)
            if "" in filter_list:
                filter_list.remove("")

            data_ = [i   for i in self.data if os.path.basename(i) in filter_list]
            self.data = data_

        self.info_label_dict = self.get_info_label_dict()
        print(self.info_label_dict)

        self.info_df = self.get_info_df(gen_list)

        self.info_dict = self.info_df.set_index('filename').T.to_dict()

        if args.balance_data and gen_list != []:
            gen_balance_list = []
            # 提取数量最多的属性，数量乘2后计算其他组需要多少生成图片达到与最多数量的组一致
            group_nums = self.info_df[(self.info_df['generate'] == False)].value_counts(
                args.balance_attribute).sort_values(
                ascending=False)
            group_nums[0] *= 1
            max_count = group_nums[0]
            balance_count = max_count - group_nums
            print("平衡数据量")
            print(balance_count)
            balance_count = balance_count.reset_index()
            for group, need_nums in balance_count.values:
                select_df = self.info_df[
                    (self.info_df['generate'] == True) & (self.info_df[args.balance_attribute] == group)]
                if len(select_df) < need_nums:
                    print(f"{args.balance_attribute} {group}生成图片数量缺少{need_nums - len(select_df)}张")
                    select_gen = select_df['path'].values
                else:
                    print(f"{args.balance_attribute} {group} 数据集已平衡")
                    select_gen = select_df.sample(need_nums)['path'].values

                gen_balance_list.append(select_gen)
            gen_list = np.concatenate(gen_balance_list)
        else:
            gen_list = gen_list

        # gen_list = gen_list[:len(self.data)]
        count = 0
        for path in gen_list:
            index = re.findall("(data_\d+)_", path)[0]
            index += ".npz"
            basename = os.path.basename(path)

            if basename in self.info_dict:
                self.data.append(path)
                count += 1
        print(f"生成数据数量: {count}")

        self.targets = []
        for path in self.data:
            basename = os.path.basename(path)
            label_name = self.info_df[self.info_df['filename'] == basename]['label'].iloc[0]

            self.targets.append(self.info_label_dict['label'][label_name])

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.class_to_idx = self.info_label_dict['label']


    def get_info_df(self,gen_list):

        new_summary_df = self.summary_data[['filename','age','gender','race','ethnicity','language','maritalstatus','label']]
        new_summary_df['generate'] = False
        new_summary_df['path'] = ""
        gen_info_list = [new_summary_df]
        for path in gen_list:
            basename = os.path.basename(path)
            index = re.findall("/(data_\d+)_",path)[0]
            index += ".npz"
            select = new_summary_df[new_summary_df['filename'] == index]
            if len(select) == 1:

                select['generate'] = True
                select['filename'] = basename
                select['path'] = path
                gen_info_list.append(select)

        new_summary_df = pd.concat(gen_info_list)




        return new_summary_df

    def get_info_label_dict(self):
        result = {}
        select_col = ['gender','race','ethnicity','language','maritalstatus']

        for col_name in select_col:
            keys = self.summary_data[col_name].unique()
            result[col_name] = {i:None for i in keys}
        for i,path in enumerate(self.data):
            basename = os.path.basename(path)
            raw_data = np.load(path, allow_pickle=True)

            info = self.summary_data[self.summary_data['filename']==basename]

            for k,v in result.items():

                result[k][info[k].iloc[0]] =raw_data[k].item()
            for k,v in result.items():
                if None in set(result[k].values()):
                    break
            else:
                break
        result["age"] = {"young": 0, 'elderly': 1}
        result["label"] = {'no':0,'yes':1}

        return result


class Dataset10k():

    def __init__(self,args,root_path,path_list,gen_list):
        filter_path, summary_path = os.path.join(root_path, "filter_file.txt"),os.path.join(root_path, "data_summary.csv")

        self.data = path_list


        self.summary_data = pd.read_csv(summary_path)

        if filter_path:
            with open(filter_path,'r') as f:
                filter_list = f.read()
                filter_list = filter_list.split("\n")

            filter_list = [i.replace(".png",".npz") for i in filter_list]
            filter_list = set(filter_list)
            if "" in filter_list:
                filter_list.remove("")

            data_ = [i   for i in self.data if os.path.basename(i) in filter_list]
            self.data = data_

        self.info_label_dict = self.get_info_label_dict()
        print(self.info_label_dict)

        self.info_df = self.get_info_df(gen_list)

        self.info_dict =self.info_df.set_index('filename').T.to_dict()

        if args.balance_data and gen_list != []:
            gen_balance_list = []
            #提取数量最多的属性，数量乘2后计算其他组需要多少生成图片达到与最多数量的组一致
            group_nums = self.info_df[(self.info_df['generate'] == False)].value_counts(args.balance_attribute).sort_values(
                ascending=False)
            group_nums[0] *= 1
            max_count = group_nums[0]
            balance_count = max_count - group_nums
            print("平衡数据量")
            print(balance_count)
            balance_count = balance_count.reset_index()
            for group,need_nums in balance_count.values:
                select_df = self.info_df[
                    (self.info_df['generate'] == True) & (self.info_df[args.balance_attribute] == group)]
                if len(select_df) < need_nums:
                    print(f"{args.balance_attribute} {group}生成图片数量缺少{need_nums - len(select_df)}张")
                    select_gen = select_df['path'].values
                else:
                    print(f"{args.balance_attribute} {group} 数据集已平衡")
                    select_gen = select_df.sample(need_nums)['path'].values


                gen_balance_list.append(select_gen)
            gen_list = np.concatenate(gen_balance_list)
        else:
            gen_list = gen_list



        # gen_list = gen_list[:len(self.data)]
        count = 0
        for path in gen_list:
            index = re.findall("(data_\d+)_", path)[0]
            index += ".npz"
            basename = os.path.basename(path)

            if basename in self.info_dict:
                self.data.append(path)
                count += 1
        print(f"生成数据数量: {count}")









        self.targets = []
        for path in self.data:
            basename = os.path.basename(path)

            label_name = self.info_df[self.info_df['filename'] == basename]['label'].iloc[0]

            self.targets.append(self.info_label_dict['label'][label_name])

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.class_to_idx = self.info_label_dict['label']

    def get_info_df(self,gen_list):

        new_summary_df = self.summary_data[['filename','age','gender','race','ethnicity','language','maritalstatus','label']]
        new_summary_df['generate'] = False
        new_summary_df['path'] = ""
        gen_info_list = [new_summary_df]
        for path in gen_list:
            basename = os.path.basename(path)
            index = re.findall("/(data_\d+)_",path)[0]
            index += ".npz"
            select = new_summary_df[new_summary_df['filename'] == index]
            if len(select) == 1:

                select['generate'] = True
                select['filename'] = basename
                select['path'] = path
                gen_info_list.append(select)

        new_summary_df = pd.concat(gen_info_list)




        return new_summary_df

    def get_info_label_dict(self):
        result = {}
        select_col = ['gender','race','ethnicity','language','maritalstatus']

        for col_name in select_col:
            keys = self.summary_data[col_name].unique()
            result[col_name] = {i:None for i in keys}
        for i,path in enumerate(self.data):
            basename = os.path.basename(path)
            raw_data = np.load(path, allow_pickle=True)

            info = self.summary_data[self.summary_data['filename']==basename]

            for k,v in result.items():

                result[k][info[k].iloc[0]] =raw_data[k].item()
            for k,v in result.items():
                if None in set(result[k].values()):
                    break
            else:
                break
        result["age"] = {"young": 0, 'elderly': 1}
        result["label"] = {'Health':0,'Glaucoma':1}

        return result













def update_class_sample_counts(dataset):
    class_counts = defaultdict(int)
    for _, target in dataset.samples:
        class_counts[target] += 1
    cls_num_list = [class_counts[class_idx] for class_idx in sorted(class_counts)]
    return cls_num_list
def get_datasets(args):
    trans = {
        # Train uses data augmentation
        'train':
            transforms.Compose([
                transforms.RandomCrop(256, padding=4),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),

                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
        # Validation does not use augmentation
        'valid':
            transforms.Compose([
                transforms.RandomCrop(256, padding=4),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),

        # Test does not use augmentation
        'test':
            transforms.Compose([
                transforms.RandomCrop(256, padding=4),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
    }
    gen_image_config = {"top_precentege":float(args.top_precentege)}
    print(f"gen_image_config :{gen_image_config}")

    data_root = args.data_root

    if args.dataset == "10k":
        root_path = os.path.join(data_root,f"10k")
        train_root_dir = os.path.join(root_path,f"Training")
        val_root_dir = os.path.join(root_path,f"Validation")
        test_root_dir = os.path.join(root_path,f"Test")
        train_root_dir = os.path.join(data_root, train_root_dir)
        val_root_dir = os.path.join(data_root, val_root_dir)
        test_root_dir = os.path.join(data_root, test_root_dir)

        train_list = glob(os.path.join(train_root_dir, "*.npz"))
        validation_list = glob(os.path.join(val_root_dir, "*.npz"))
        test_list = glob(os.path.join(test_root_dir, "*.npz"))

        if args.use_fake_data:
            metric_path = os.path.join(data_root,'gen_image',args.dataset,"sd_gen_metric.csv")
            metric = pd.read_csv(metric_path)
            # 计算排名并综合排序
            metric["dice_rank"] = metric["dice"].rank(ascending=False)
            metric["hd95_rank"] = metric["hd95"].rank(ascending=True)
            metric["combined_rank"] = metric["dice_rank"] + metric["hd95_rank"]
            metric = metric.sort_values("combined_rank")
            top_n = metric.head(int(len(metric) * gen_image_config['top_precentege']))
            gen_list = top_n['case_name'].values
            gen_list = [os.path.join(data_root, 'gen_image',args.dataset, "sd_gen0", i + ".png") for i in gen_list]
            # gen_list =glob(os.path.join(data_root,'gen_image',"sd_gen0", "*.png"))
            # gen_list =[i for i in gen_list if "generate" in i]
        else:
            gen_list = []

        # random.shuffle(gen_list)
        len_gen_list = len(gen_list)
        train_num = int(len_gen_list * 0.7)
        val_num = int(len_gen_list * 0.1)
        gen_train_list = gen_list
        gen_validation_list = []
        gen_test_list = []



        # Generators
        training_dataset = Dataset10k(args,root_path,train_list,gen_train_list, )
        validation_dataset = Dataset10k(args,root_path,validation_list,gen_validation_list )
        test_dataset = Dataset10k(args,root_path,test_list,gen_test_list)

        training_dataset = mydatasets.WrapperDataset(data_class=training_dataset,
                                                     transform=trans['train'], task='multi-class')
        validation_dataset = mydatasets.WrapperDataset(data_class=validation_dataset,
                                                       transform=trans['valid'], task='multi-class')
        test_dataset = mydatasets.WrapperDataset(data_class=test_dataset, transform=trans['test'],
                                                 task='multi-class')
    elif args.dataset == 'fairvlmed10k':
        root_path = os.path.join(data_root, f"fairvlmed10k")
        train_root_dir = os.path.join(root_path, f"Training")
        val_root_dir = os.path.join(root_path, f"Validation")
        test_root_dir = os.path.join(root_path, f"Test")
        train_root_dir = os.path.join(data_root, train_root_dir)
        val_root_dir = os.path.join(data_root, val_root_dir)
        test_root_dir = os.path.join(data_root, test_root_dir)

        train_list = glob(os.path.join(train_root_dir, "*.npz"))
        validation_list = glob(os.path.join(val_root_dir, "*.npz"))
        test_list = glob(os.path.join(test_root_dir, "*.npz"))

        if args.use_fake_data:
            metric_path = os.path.join(data_root,'gen_image',args.dataset,"sd_gen_metric.csv")
            metric = pd.read_csv(metric_path)
            # 计算排名并综合排序
            metric["dice_rank"] = metric["dice"].rank(ascending=False)
            metric["hd95_rank"] = metric["hd95"].rank(ascending=True)
            metric["combined_rank"] = metric["dice_rank"] + metric["hd95_rank"]
            metric = metric.sort_values("combined_rank")
            top_n = metric.head(int(len(metric) * gen_image_config['top_precentege']))
            gen_list = top_n['case_name'].values
            gen_list = [os.path.join(data_root, 'gen_image',args.dataset, "sd_gen0", i + ".png") for i in gen_list]
            # gen_list =glob(os.path.join(data_root,'gen_image',"sd_gen0", "*.png"))
            # gen_list =[i for i in gen_list if "generate" in i]
        else:
            gen_list = []

        # random.shuffle(gen_list)
        gen_train_list = gen_list






        # Generators
        training_dataset = DatasetFairvlmed10k(args,root_path,train_list,gen_train_list)
        validation_dataset = DatasetFairvlmed10k(args,root_path,validation_list,gen_train_list)
        test_dataset = DatasetFairvlmed10k(args,root_path,test_list,gen_train_list)

        training_dataset = mydatasets.WrapperDataset(data_class=training_dataset,
                                                     transform=trans['train'], task='multi-class')
        validation_dataset = mydatasets.WrapperDataset(data_class=validation_dataset,
                                                       transform=trans['valid'], task='multi-class')
        test_dataset = mydatasets.WrapperDataset(data_class=test_dataset, transform=trans['test'],
                                                 task='multi-class')
    else:
        raise


    return training_dataset, test_dataset, validation_dataset


def multiclass_equalized_odds(pred_prob, y, attrs):
    pred_one_hot = prob_to_label(pred_prob)

    gt_one_hot = numeric_to_one_hot(y)

    scores = []
    for i in range(pred_one_hot.shape[1]):
        tmp_score = equalized_odds_difference(pred_one_hot[:, i],
                                              gt_one_hot[:, i],
                                              sensitive_features=attrs)

        scores.append(tmp_score)

    avg_score = np.mean(scores)

    return avg_score

def prob_to_label(pred_prob):
    # Find the indices of the highest probabilities for each sample
    max_prob_indices = np.argmax(pred_prob, axis=1)

    # Create one-hot vectors for each sample
    one_hot_vectors = np.zeros_like(pred_prob)
    one_hot_vectors[np.arange(len(max_prob_indices)), max_prob_indices] = 1

    return one_hot_vectors


def numeric_to_one_hot(y, num_classes=None):
    y = np.asarray(y, dtype=np.int32)

    if num_classes is None:
        num_classes = np.max(y) + 1

    one_hot_array = np.zeros((len(y), num_classes))
    one_hot_array[np.arange(len(y)), y] = 1

    return one_hot_array
def multiclass_demographic_parity(pred_prob, y, attrs):
    pred_one_hot = prob_to_label(pred_prob)

    gt_one_hot = numeric_to_one_hot(y)

    scores = []
    for i in range(pred_one_hot.shape[1]):
        tmp_score = demographic_parity_difference(pred_one_hot[:, i],
                                                  gt_one_hot[:, i],
                                                  sensitive_features=attrs)

        scores.append(tmp_score)

    avg_score = np.mean(scores)

    return avg_score
def compute_between_group_disparity(auc_list, overall_auc):
    return np.std(auc_list) / overall_auc, (np.max(auc_list)-np.min(auc_list)) / overall_auc
def equity_scaled_AUC(output, target, attrs, alpha=1., num_classes=2):
    es_auc = 0
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []


    y_onehot = num_to_onehot(target, num_classes)
    overall_auc = roc_auc_score(y_onehot, output, average='macro', multi_class='ovr')

    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]


        y_onehot = num_to_onehot(gt_group, num_classes)
        try:
            group_auc = roc_auc_score(y_onehot, pred_group, average='macro', multi_class='ovr')
        except:
            group_auc = np.nan
        identity_wise_perf.append(group_auc)
        identity_wise_num.append(gt_group.shape[0])

    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i] - overall_auc)
    es_auc = (overall_auc / (alpha * tmp + 1))

    return es_auc
def equity_scaled_accuracy(output, target, attrs, alpha=1.):
    es_acc = 0
    if len(output.shape) >= 2:
        overall_acc = np.sum(np.argmax(output, axis=1) == target) / target.shape[0]
    else:
        overall_acc = np.sum((output >= 0.5).astype(float) == target) / target.shape[0]
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []
    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]

        if len(pred_group.shape) >= 2:
            acc = np.sum(np.argmax(pred_group, axis=1) == gt_group) / gt_group.shape[0]
        else:
            acc = np.sum((pred_group >= 0.5).astype(float) == gt_group) / gt_group.shape[0]

        identity_wise_perf.append(acc)
        identity_wise_num.append(gt_group.shape[0])

    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i] - overall_acc)
    es_acc = (overall_acc / (alpha * tmp + 1))

    return es_acc

def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if len(output.shape) == 1:
        acc = np.sum((output >= 0.5).astype(float) == target) / target.shape[0]
        return acc.item()
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, dim=1)
        target = target.view(batch_size, 1).repeat(1, maxk)

        correct = (pred == target)

        topk_accuracy = []
        for k in topk:
            accuracy = correct[:, :k].float().sum().item()
            accuracy /= batch_size  # [0, 1.]
            topk_accuracy.append(accuracy)

        return topk_accuracy[0]

def num_to_onehot(nums, num_to_class):
    nums = nums.astype(int)
    n_values = num_to_class
    onehot_vec = np.eye(n_values)[nums]
    return onehot_vec
def compute_auc(pred_prob, y, num_classes=2):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()


    y_onehot = num_to_onehot(y, num_classes)
    try:
        auc_val = roc_auc_score(y_onehot, pred_prob, average='macro', multi_class='ovr')
    except:
        auc_val = np.nan

    return auc_val

def evalute_comprehensive_perf(preds, gts, attrs=None, num_classes=2):

    esaccs_by_attrs = []
    esaucs_by_attrs = []
    aucs_by_attrs = []
    dpds = []
    dprs = []
    eods = []
    eors = []
    between_group_disparity = []
    overall_auc = compute_auc(preds, gts, num_classes=num_classes)
    # 计算混淆矩阵
    pred_labels = np.argmax(preds,axis=1)
    tn, fp, fn, tp = confusion_matrix(gts, pred_labels).ravel()
    # 计算specificity
    specificity = tn / (tn + fp)
    # 计算sensitivity
    sensitivity = tp / (tp + fn)
    # 计算precision
    precision = tp / (tp + fp)
    # 计算F1 score
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

    for i in range(attrs.shape[0]):
        attr = attrs[i,:]

        es_acc = equity_scaled_accuracy(preds, gts, attr)
        esaccs_by_attrs.append(es_acc)
        es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        esaucs_by_attrs.append(es_auc)

        aucs_by_group = []
        elements = np.unique(attr).astype(int)
        for e in elements:
            aucs_by_group.append( compute_auc(preds[attr == e], gts[attr == e], num_classes=num_classes) )
        aucs_by_attrs.append(aucs_by_group)
        std_disparity, max_disparity = compute_between_group_disparity(aucs_by_group, overall_auc)
        between_group_disparity.append([std_disparity, max_disparity])

        if num_classes == 2:
            dpd = demographic_parity_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            dpr = demographic_parity_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eod = equalized_odds_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eor = equalized_odds_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
        elif num_classes > 2:
            dpd = multiclass_demographic_parity(preds, gts, attr)
            dpr = 0
            eod = multiclass_equalized_odds(preds, gts, attr)
            eor = 0

        dpds.append(dpd)
        eods.append(eod)

    return esaccs_by_attrs, overall_auc, esaucs_by_attrs, aucs_by_attrs, dpds, eods, between_group_disparity,specificity,sensitivity,f1,precision


def count_number_of_groups(input_dataset):
    instances_on_race = []
    instances_on_gender = []
    instances_on_ethnicity = []
    for file in input_dataset.files:
            npz_path = file
            data = np.load(npz_path)
            instances_on_race.append(data['race'].item())
            instances_on_gender.append(data['gender'].item())
            instances_on_ethnicity.append(data['ethnicity'].item())
    # count the unique number in instances_on_race
    _, numbers_of_race = np.unique(instances_on_race, return_counts=True)
    _, numbers_of_gender = np.unique(instances_on_gender, return_counts=True)
    _, numbers_of_ethnicity = np.unique(instances_on_ethnicity, return_counts=True)
    return numbers_of_race, numbers_of_gender, numbers_of_ethnicity