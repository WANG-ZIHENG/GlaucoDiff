import gc
import os
import re
import shutil
import uuid
from glob import glob

import einops
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from cldm.ddim_hacked import DDIMSampler
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from share import *
from tutorial_dataset import MyDataset, collate_process


def get_last_lightning_cpkpt(result_dir, delete_old=False):
    files_names = glob(os.path.join(result_dir, 'lightning_logs/*'))
    latest_file = sorted(files_names, key=lambda x: x.split("_")[-1], reverse=True)
    for file in latest_file:
        ckpt_path = os.path.join(file, "checkpoints", "*")
        ckpt_path = glob(ckpt_path)
        if len(ckpt_path) == 0:
            continue
        ckpt_path = ckpt_path[0]
        if delete_old:
            latest_file.remove(file)
            for i in latest_file:
                shutil.rmtree(i)
        return ckpt_path
    raise FileNotFoundError("No checkpoints found")


def get_last_global(result_dir):
    log_file = os.path.join(result_dir, 'log_train.txt')
    last_global = 0
    if not os.path.exists(log_file):
        return last_global
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if 'fid:' in line:
                parts = line.split(' fid:')
                if len(parts) >= 1:
                    try:
                        epoch_num = int(parts[0].strip())
                        last_global = max(last_global, epoch_num)
                    except ValueError:
                        continue
    return last_global


def train_control(args, global_epoch, topn_file, created_model):
    resume_path = './models/control_sd21_ini.ckpt'
    batch_size = 1
    logger_freq = 1000
    learning_rate = 5e-5
    sd_locked = False
    only_mid_control = False
    finetune_epoch = 1

    model = created_model
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    if global_epoch == 0:
        model.load_state_dict(load_state_dict(resume_path, location='cpu'))
        ckpt_path = None
        now_epoch = -1
    else:
        ckpt_path = get_last_lightning_cpkpt(args.result_dir)
        now_epoch = int(re.findall(r'epoch=(\d+)', ckpt_path)[0])
    max_epoch = now_epoch + finetune_epoch + 1

    dataset = MyDataset(args=args, file_list=topn_file)
    dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_process)
    image_logger = ImageLogger(batch_frequency=logger_freq,
                               log_images_kwargs={"ddim_steps": 25}, disabled=False)
    save_checkpoint = pl.callbacks.ModelCheckpoint(every_n_epochs=finetune_epoch, save_top_k=1)
    trainer = pl.Trainer(gpus=1, precision=32,
                         callbacks=[image_logger, save_checkpoint],
                         max_epochs=max_epoch, weights_save_path=args.result_dir)
    trainer.fit(model, dataloader, ckpt_path=ckpt_path)
    del trainer, model, dataloader, image_logger, dataset
    gc.collect()
    torch.cuda.empty_cache()


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution,
            detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta,
            model, ddim_sampler):
    with torch.no_grad():
        input_image = input_image.squeeze()
        H, W, C = input_image.shape

        control = input_image
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = np.random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control],
                "c_crossattn": [model.get_learned_conditioning(
                    [prompt[0] + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control],
                   "c_crossattn": [model.get_learned_conditioning(
                       [n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = ([strength * (0.825 ** float(12 - i)) for i in range(13)]
                                if guess_mode else ([strength] * 13))
        samples, _ = ddim_sampler.sample(ddim_steps, num_samples, shape, cond,
                                         verbose=False, eta=eta,
                                         unconditional_guidance_scale=scale,
                                         unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c')
                     * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        results = [x_samples[i] for i in range(num_samples)]
    return [input_image.permute(2, 0, 1).cpu()] + results


def gen_new_image(args, topn_file, created_model, global_epoch=0,
                  ckpt_path=None, ddim_steps=30, save_dir=None):
    model = created_model
    if ckpt_path is None:
        ckpt_path = get_last_lightning_cpkpt(args.result_dir)
    model.load_state_dict(load_state_dict(ckpt_path, location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    dataset = MyDataset(args=args, gen_data=True, file_list=topn_file, gen_scale_masks=True)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=False,
                            collate_fn=collate_process)

    if save_dir is None:
        save_dir = os.path.join(args.result_dir, f"sd_gen_step_{ddim_steps}_{global_epoch}")
    os.makedirs(save_dir, exist_ok=True)

    for i, item in enumerate(tqdm(dataloader)):
        extra_masks = item['extra_mask']
        relativs = item['relativs']
        clahe_target = item['clahe_target']
        prompt = item['txt']
        source = item['hint'].to('cuda')
        output_name = item['name'][0]
        generated_uuid = str(uuid.uuid4())[:4]
        output = process(source, prompt, '', '', 1, 512, 512,
                         ddim_steps=ddim_steps, guess_mode=False,
                         strength=1.0, scale=9.0,
                         seed=int(np.random.randint(0, 65536)), eta=0,
                         model=model, ddim_sampler=ddim_sampler)

        for j in range(len(output)):
            img = output[j]
            img = transforms.ToPILImage()(img)
            if j == 0:
                target_img = clahe_target[0].permute(2, 0, 1)
                target_img = transforms.ToPILImage()(target_img)
                origin_mask_proportion = relativs[0][-1]
                if not os.path.exists(os.path.join(save_dir, f'{output_name}_target.png')):
                    target_img.save(os.path.join(save_dir, f'{output_name}_target.png'))
                    img.save(os.path.join(save_dir,
                             f'{output_name}_{generated_uuid}_{origin_mask_proportion}.png'))
                    with open(os.path.join(save_dir, f'{output_name}_prompt.txt'), 'w') as f:
                        f.write(prompt[0])
            else:
                png_path = os.path.join(save_dir,
                                        f'{output_name}_{generated_uuid}_{ddim_steps}_generate.png')
                img = img.convert('L')
                img.save(png_path)

        for extra_mask, relativ in zip(extra_masks[0][:-1], relativs[0][:-1]):
            extra_mask = torch.tensor(extra_mask[np.newaxis, :, :]).to('cuda').to(torch.float)
            output = process(extra_mask, prompt, '', '', 1, 512, 512,
                             ddim_steps=30, guess_mode=False,
                             strength=1.0, scale=9.0,
                             seed=int(np.random.randint(0, 65536)), eta=0,
                             model=model, ddim_sampler=ddim_sampler)
            extra_mask_img = extra_mask[0].permute(2, 0, 1)
            extra_mask_img = transforms.ToPILImage()(extra_mask_img)
            extra_mask_img.save(os.path.join(save_dir,
                                f'{output_name}_{generated_uuid}_extra_{relativ}.png'))
            for j in range(len(output)):
                if j != 0:
                    img = output[j]
                    img = transforms.ToPILImage()(img)
                    png_path = os.path.join(
                        save_dir,
                        f'{output_name}_{generated_uuid}_extra_{relativ}_{ddim_steps}_generate.png')
                    img = img.convert('L')
                    img.save(png_path)

    del model, dataset, dataloader, ddim_sampler
    gc.collect()
    torch.cuda.empty_cache()
