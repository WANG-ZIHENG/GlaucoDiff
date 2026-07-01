import copy
import os
import random
from glob import glob

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from scipy.optimize import minimize
from shapely.geometry import Polygon
from skimage import measure
from torch.utils.data import Dataset


def random_crop(image, start_h, start_w, crop_height, crop_width):
    if image.shape[0] < crop_height or image.shape[1] < crop_width:
        raise ValueError("Image dimensions should be larger than crop dimensions")
    return image[start_h:start_h + crop_height, start_w:start_w + crop_width]


def collate_process(x):
    t = x[0]
    output_dict = {}
    exclude = {"relativs", "extra_mask", "prompts"}
    for key, item in t.items():
        if key not in exclude and not isinstance(item, str):
            output_dict[key] = torch.tensor(np.array([i[key] for i in x]))
        else:
            output_dict[key] = [i[key] for i in x]
    return output_dict


class MyDataset(Dataset):
    def __init__(self, args, gen_data=False, file_list=None, gen_scale_masks=False):
        self.data = []
        self.gen_scale_masks = gen_scale_masks
        self.args = args
        summary = os.path.join(args.dataset_dir, 'data_summary.csv')
        self.summary_data = pd.read_csv(summary).set_index('filename')
        self.gen_data = gen_data
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.palette = {0: (0, 0, 0),
                        -2: (255, 0, 0),
                        -1: (0, 0, 255),
                        3: (0, 255, 0)}

        if file_list is not None:
            self.data = file_list
        else:
            self.data = glob(os.path.join(args.dataset_dir, 'Training', "*.npz"))

        if args.use_filter:
            filter_path = os.path.join(args.dataset_dir, 'filter_file.txt')
            with open(filter_path, 'r') as f:
                filter_list = f.read().split("\n")
            filter_list = [i.replace(".png", ".npz") for i in filter_list]
            filter_list = set(filter_list)
            filter_list.discard("")
            self.data = [i for i in self.data if os.path.basename(i) in filter_list]

    def __len__(self):
        return len(self.data)

    def check_contact(self, mask, max_min_distance=4):
        blue_mask = (mask == -2).astype(np.uint8)
        red_mask = np.logical_or(mask == -1, mask == -2).astype(np.uint8)
        red_contours = measure.find_contours(red_mask, 0.99)[0]
        blue_contours = measure.find_contours(blue_mask, 0.99)[0]

        mask1_polygon = Polygon(red_contours)
        mask2_polygon = Polygon(blue_contours)
        distance = mask1_polygon.exterior.distance(mask2_polygon.exterior)
        if mask2_polygon.within(mask1_polygon) and distance > max_min_distance:
            return False
        return True

    def cal_ellipse(self, mask, origin_mask):
        contours = measure.find_contours(mask, 0.99)[0]
        ellipse = cv2.fitEllipse(contours.reshape(-1, 1, 2).astype(int))
        mask = np.ascontiguousarray(mask)
        out = cv2.ellipse(mask, ellipse, (3), 2)
        _, cols = np.where(out == 3)
        return int(np.max(cols) - np.min(cols)), origin_mask

    def cal_proportion(self, mask):
        red_mask = np.logical_or(mask == -1, mask == -2).astype(np.uint8)
        blue_mask = (mask == -2).astype(np.uint8)
        red_long, _ = self.cal_ellipse(red_mask, mask)
        blue_long, _ = self.cal_ellipse(blue_mask, mask)
        relative_size = blue_long / red_long
        return round(relative_size, 2), mask

    def scale_bule_mask(self, mask_image, scale_factor):
        red_mask = np.logical_or(mask_image == -1, mask_image == -2).astype(np.uint8)
        blue_mask = (mask_image == -2).astype(np.uint8)
        zero_mask = np.logical_or(mask_image == 0, mask_image == None)
        rows, cols = np.where(blue_mask > 0)
        blue_center = (np.mean(rows), np.mean(cols))
        top, bottom = np.min(rows), np.max(rows)
        left, right = np.min(cols), np.max(cols)
        blue_mask_scaled = ndimage.zoom(blue_mask[top:bottom + 1, left:right + 1], scale_factor, order=0)
        blue_mask_scaled = blue_mask_scaled.astype(np.float64)
        blue_mask_scaled[blue_mask_scaled == 1] = -2
        blue_mask_scaled[blue_mask_scaled == 0] = -1
        if scale_factor < 1:
            new_top = top + (bottom - top + 1) * (1 - scale_factor) // 2
            new_bottom = bottom - (bottom - top + 1) * (1 - scale_factor) // 2
            new_left = left + (right - left + 1) * (1 - scale_factor) // 2
            new_right = right - (right - left + 1) * (1 - scale_factor) // 2
        else:
            new_top = top - (bottom - top + 1) * (scale_factor - 1) // 2
            new_bottom = bottom + (bottom - top + 1) * (scale_factor - 1) // 2
            new_left = left - (right - left + 1) * (scale_factor - 1) // 2
            new_right = right + (right - left + 1) * (scale_factor - 1) // 2
        new_center = (new_top + (new_bottom - new_top) / 2, new_left + (new_right - new_left) / 2)
        offset_row = int(blue_center[0] - new_center[0])
        offset_col = int(blue_center[1] - new_center[1])
        new_top = int(new_top)
        new_left = int(new_left)

        scaled_mask = np.zeros_like(mask_image)
        scaled_mask[red_mask > 0] = -1
        scaled_mask[new_top + offset_row:new_top + offset_row + blue_mask_scaled.shape[0],
                    new_left + offset_col:new_left + offset_col + blue_mask_scaled.shape[1]] = blue_mask_scaled
        scaled_mask[zero_mask] = 0
        return scaled_mask

    def solve(self, mask_image, target_relative_size):
        def solve_func(params, mask_image, target_relative_size):
            [scale] = params
            scaled_mask = self.scale_bule_mask(mask_image, scale)
            try:
                relative_size, _ = self.cal_proportion(scaled_mask)
            except Exception:
                relative_size = 10
            return abs(target_relative_size - relative_size)

        initial_guess = [1]
        result = minimize(solve_func, initial_guess,
                          args=(mask_image, target_relative_size),
                          method='nelder-mead', tol=1e-3, bounds=[(0.1, 3)])
        [relative_size] = result.x
        return relative_size

    def get_scale_mask(self, mask_image):
        mask_relative_size, _ = self.cal_proportion(mask_image)
        scale_factors = np.arange(-0.8, 2, 0.05)
        scaled_masks = []
        relativs = []
        for scale_factor in scale_factors:
            try:
                if scale_factor == mask_relative_size:
                    continue
                scale = self.solve(mask_image, target_relative_size=mask_relative_size + scale_factor)
                scaled_mask = self.scale_bule_mask(mask_image, scale)
                relative_size, scaled_mask = self.cal_proportion(scaled_mask)
                try:
                    contact = self.check_contact(scaled_mask)
                except Exception:
                    contact = True

                if 0.2 <= relative_size <= 0.9 and contact is False:
                    scaled_masks.append(scaled_mask)
                    relativs.append(relative_size)
                elif contact is True or relative_size > 0.9:
                    break
            except Exception:
                continue
        return scaled_masks, relativs

    def getitem(self, idx):
        data_file = self.data[idx]
        raw_data = np.load(data_file, allow_pickle=True)
        modified_image = raw_data['slo_fundus'].astype(np.uint8)
        modified_image = np.array([modified_image, modified_image, modified_image]).transpose(1, 2, 0)
        index = os.path.basename(data_file)
        info = self.summary_data.loc[index]
        clahe_img = modified_image

        if 'fairvlmed10k' in self.args.dataset_dir:
            mask_path = os.path.join(self.args.dataset_dir, "mask",
                                     index.replace(".npz", "_predict.png"))
            mask_image = Image.open(mask_path)
            mask_image = np.array(mask_image).astype(float)
            mask_image[mask_image == 128] = -1
            mask_image[mask_image == 255] = -2
        else:
            mask_image = raw_data['disc_cup_mask']

        if self.gen_scale_masks:
            scaled_masks, relativs = self.get_scale_mask(mask_image)
            origin_mask_proportion, mask_image = self.cal_proportion(mask_image)
            prompts = ['Age ' + str(info['age']) + ', ' + ', '.join(
                info.iloc[1:4].values.tolist()) +
                f",Speaking {info['language']},{info.iloc[5]}" +
                f",Cup-to-Disc Ratio {i}" for i in relativs]
            scaled_masks.append(mask_image)
            relativs.append(origin_mask_proportion)
        else:
            origin_mask_proportion, mask_image = self.cal_proportion(mask_image)
            scaled_masks, relativs = [], []
            prompts = []

        prompt = ('Age ' + str(info['age']) + ', ' +
                  ', '.join(info.iloc[1:4].values.tolist()) +
                  f",Speaking {info['language']},{info.iloc[5]}" +
                  f",Cup-to-Disc Ratio {origin_mask_proportion}")
        prompts.append(prompt)

        clahe_img_mask = copy.deepcopy(clahe_img)
        clahe_img_mask[mask_image == -1.0] = self.palette[-1]
        clahe_img_mask[mask_image == -2.0] = self.palette[-2]
        clahe_img_mask[mask_image == 3] = self.palette[3]

        source = clahe_img_mask
        target = clahe_img

        crop_height = crop_width = 512
        start_h = (source.shape[0] - crop_height + 1) // 2
        start_w = (source.shape[1] - crop_width + 1) // 2
        source = random_crop(source, start_h, start_w, crop_height, crop_width)
        target = random_crop(target, start_h, start_w, crop_height, crop_width)

        for i in range(len(scaled_masks)):
            scaled_mask = scaled_masks[i]
            target_img_copy = copy.deepcopy(clahe_img)
            target_img_copy[scaled_mask == -1.0] = self.palette[-1]
            target_img_copy[scaled_mask == -2.0] = self.palette[-2]
            target_img_copy = target_img_copy.astype(np.uint8)
            target_img_copy = random_crop(target_img_copy, start_h, start_w, crop_height, crop_width)
            target_img_copy = cv2.cvtColor(target_img_copy, cv2.COLOR_BGR2RGB)
            target_img_copy = target_img_copy.astype(np.float32) / 255.0
            scaled_masks[i] = target_img_copy

        if not self.gen_data:
            if random.randint(1, 100) >= 50:
                source = cv2.flip(source, 1)
                target = cv2.flip(target, 1)
            if random.randint(1, 100) >= 50:
                source = cv2.flip(source, 0)
                target = cv2.flip(target, 0)

        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        clahe_target = copy.deepcopy(target)

        source = source.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0
        name = os.path.basename(data_file).replace(".npz", "")
        glaucoma_label = 0
        race = int(raw_data['race'].item())
        gender = int(raw_data['gender'].item())
        ethnicity = int(raw_data['ethnicity'].item())
        language = int(raw_data['language'].item())

        return dict(jpg=target, txt=prompt, hint=source, name=name,
                    glaucoma_label=glaucoma_label, race=race, gender=gender,
                    ethnicity=ethnicity, language=language,
                    clahe_target=clahe_target, extra_mask=scaled_masks,
                    relativs=relativs, prompts=prompts)

    def __getitem__(self, idx):
        while True:
            try:
                return self.getitem(idx)
            except Exception:
                idx = random.randint(0, len(self) - 1)
