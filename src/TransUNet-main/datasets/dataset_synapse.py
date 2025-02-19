import copy
import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import torch.utils.data as data
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
def collate_process(x):
    t = x[0]
    output_dict = {}
    exclude = {}
    for key,item in t.items():
        if key  not in exclude and not isinstance(item,str):

            output_dict[key] = torch.tensor(np.array([i[key] for i in x]))

        else:
            output_dict[key] = [i[key] for i in x]

    return output_dict
def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        raise(RuntimeError('No Module named accimage'))
    else:
        return pil_loader(path)

class RandomGenerator(object):
    def __init__(self, output_size,mode='train'):
        self.output_size = output_size
        self.mode = mode

    def random_crop(self,image, start_h, start_w, crop_height, crop_width):
        if image.shape[0] < crop_height or image.shape[1] < crop_width:
            raise ValueError("Image dimensions should be larger than crop dimensions")
        cropped_image = image[start_h:start_h + crop_height, start_w:start_w + crop_width]

        return cropped_image

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.mode=='train':
            if random.random() > 0.5:
                image, label = random_rot_flip(image, label)
            elif random.random() > 0.5:
                image, label = random_rotate(image, label)
            x, y = image.shape
            if x != self.output_size[0] or y != self.output_size[1]:
                image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
                label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        elif self.mode=="test":
            pass
            # crop_height , crop_width = self.output_size
            # start_h = (image.shape[0] - crop_height + 1) // 2
            # start_w = (image.shape[1] - crop_width + 1) // 2
            # image = self.random_crop(image, start_h, start_w, crop_height, crop_width)
            # label = self.random_crop(label, start_h, start_w, crop_height, crop_width)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
class Dataset10k():

    def __init__(self,args,path_list):
        self.data = path_list

        summary_path = os.path.join(args.root_path,args.dataset,'data_summary.csv')
        filter_path = os.path.join(args.root_path,args.dataset,'filter_file.txt')
        self.summary_data = pd.read_csv(summary_path).set_index('filename')
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
        id_path_dict = {os.path.basename(i): i for i in self.data}
        self.data = np.array(self.data)
class dataset_sd_gen():

    def __init__(self,args,path_list):
        self.data = path_list
        self.data = np.array(self.data)

class Datasetfairvlmed10k():

    def __init__(self,args,path_list):
        self.data = path_list
        summary_path = os.path.join(args.volume_path,args.dataset,'data_summary.csv')
        filter_path = os.path.join(args.volume_path,args.dataset,'filter_file.txt')

        self.summary_data = pd.read_csv(summary_path).set_index('filename')
        if filter_path:
            with open(filter_path,'r') as f:
                filter_list = f.read()
                filter_list = filter_list.split("\n")

            filter_list = [i.replace(".png",".npz") for i in filter_list]
            filter_list = set(filter_list)
            filter_list.remove("")

            data_ = [i   for i in self.data if os.path.basename(i) in filter_list]
            self.data = data_
        id_path_dict = {os.path.basename(i): i for i in self.data}
        self.data = np.array(self.data)


class WrapperDataset(data.Dataset):
    """Dataset class for the Imagenet-LT dataset."""

    def __init__(self, data_class, transform,dataset,loader=default_loader,target_transform=None):
        self.dataset = dataset
        self.data_class = data_class
        self.transform = transform
        self.data = self.data_class.data
        self.target_transform = target_transform
        self.loader = loader
        self.samples = self.data
        self.palette = {0: (0),
           -2: (255),  # 红色
           -1: (128),# 蓝色
          }
        self.red_pix = [255, 0, 0]
        self.blue_pix = [0, 0, 255]


    def __len__(self):
        return len(self.samples)


    def __getitem__(self,index):
        path = self.samples[index]

        if self.dataset == 'fairvlmed10k':
            raw_data = np.load(path, allow_pickle=True)
            image = raw_data['slo_fundus']
            label = np.zeros_like(image)
        elif self.dataset=='10k':
            raw_data = np.load(path, allow_pickle=True)
            image = raw_data['slo_fundus']

            label = raw_data['disc_cup_mask']
            label = abs(label)
        elif self.dataset=='sd_gen0':
            image = Image.open(path)
            image = np.array(image)

            label_path = path.replace("_generate","")

            label = Image.open(label_path)
            label = np.array(label)



            condition = (label[:, :, 0] == self.blue_pix[0]) & (label[:, :, 1] == self.blue_pix[1]) & (label[:, :, 2] == self.blue_pix[2])
            label[condition] = [1, 1, 1]
            condition = (label[:, :, 0] == self.red_pix[0]) & (label[:, :, 1] == self.red_pix[1]) & (
                        label[:, :, 2] == self.red_pix[2])
            label[condition] = [2, 2, 2]
            label = label[:,:,0]

            not_mark_indxe = np.logical_and(label != 1, label != 2)
            label[not_mark_indxe] = 0


        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = ".".join(os.path.basename(path).split(".")[:-1])

        return sample


