import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.datasets import DatasetFolder
import torch.utils.data as data
from PIL import Image
import random
import os
import cv2
import random
from glob import glob
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
def random_crop(image,start_h,start_w, crop_height, crop_width):
    if image.shape[0] < crop_height or image.shape[1] < crop_width:
        raise ValueError("Image dimensions should be larger than crop dimensions")
    cropped_image = image[start_h:start_h+crop_height, start_w:start_w+crop_width]

    return cropped_image
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

class ImageData(DatasetFolder):
    def __init__(
            self,
            root: str,
            transform = None,
            target_transform  = None,
            loader = default_loader,
            is_valid_file = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
    
    def __len__(self):
        return len(self.samples)

class WrapperDataset(data.Dataset):
    """Dataset class for the Imagenet-LT dataset."""

    def __init__(self, data_class, transform,task,loader=default_loader,target_transform=None):

        self.data_class = data_class
        self.transform = transform
        self.data = self.data_class.data
        self.targets = self.data_class.targets
        self.info_dict = self.data_class.info_dict
        self.info_label_dict = self.data_class.info_label_dict
        self.task = task
        self.class_to_idx = self.data_class.class_to_idx
        self.target_transform = target_transform
        self.loader = loader
        self.samples = list(zip(self.data,self.targets))
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))





        # 统计类数量
        classes_num = {}
        id_to_class = {v: k for k, v in self.class_to_idx.items()}
        for i in self.targets:
            class_name = id_to_class[i]
            if class_name not in classes_num:
                classes_num[class_name] = 1
            else:
                classes_num[class_name] += 1


        self.cls_num_list = list(classes_num.values())
        self.class_names = list(classes_num.keys())
        self.id_to_name = id_to_class
    def __len__(self):
        return len(self.samples)


    def __getitem__(self,index):
        path, target = self.samples[index]
        basename = os.path.basename(path)
        if ".npz" in path:
            raw_data = np.load(path, allow_pickle=True)
            modified_image = raw_data['slo_fundus']
            modified_image = modified_image.astype(np.uint8)
            modified_image = np.array([modified_image, modified_image, modified_image]).transpose(1, 2, 0)
            clahe_img = np.zeros_like(modified_image)
            for i in range(3):
                clahe_img[:, :, i] = self.clahe.apply(modified_image[:, :, i])

            crop_height = crop_width = 512
            # start_h = np.random.randint(0, modified_image.shape[0] - crop_height + 1)
            # start_w = np.random.randint(0, modified_image.shape[1] - crop_width + 1)
            start_h = (modified_image.shape[0] - crop_height + 1)//2
            start_w = (modified_image.shape[1] - crop_width + 1) // 2
            clahe_img = random_crop(clahe_img, start_h, start_w, crop_height, crop_width)
            clahe_img = Image.fromarray(clahe_img)
        elif '.png' in path:
            clahe_img = self.loader(path)

        if self.transform is not None:
            clahe_img = self.transform(clahe_img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # extract glaucoma label from npz file
        race = self.info_dict[basename]['race']
        race = self.info_label_dict['race'][race]
        gender = self.info_dict[basename]['gender']
        gender = self.info_label_dict['gender'][gender]
        ethnicity = self.info_dict[basename]['ethnicity']
        ethnicity = self.info_label_dict['ethnicity'][ethnicity]
        language = self.info_dict[basename]['language']
        language = self.info_label_dict['language'][language]
        maritalstatus = self.info_dict[basename]['maritalstatus']
        maritalstatus = self.info_label_dict['maritalstatus'][maritalstatus]
        age = self.info_dict[basename]['age']
        age = self.info_label_dict['age'][age]

        # merge all labels together into a single tensor at size of 4
        label_and_attributes = torch.tensor([ race, gender, ethnicity, language,maritalstatus,age])

        return dict(image= clahe_img,target=target,label_and_attributes=label_and_attributes)
