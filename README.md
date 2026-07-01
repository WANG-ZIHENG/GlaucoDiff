# GlaucoDiff

Official code for **Fairness-Aware vCDR-Controlled Generation for Glaucoma Diagnosis** (MICCAI 2025).

[Paper](https://papers.miccai.org/miccai-2025/paper/1939_paper.pdf)

GlaucoDiff is a Stable Diffusion 2.1 + ControlNet pipeline that synthesises
SLO fundus images with precise control over the vertical cup-to-disc ratio
(vCDR). Given an SLO image and a paired optic-cup / optic-disc mask, GlaucoDiff
overlays the mask on the source image and generates a new SLO image whose
vCDR matches a demographically-conditioned text prompt.

## Installation

```bash
conda create -n GlaucoDiff python=3.10
conda activate GlaucoDiff
pip install -r requirements.txt
```

## Data preparation

We use two public SLO fundus datasets:

- Harvard-FairSeg (referred to as `10k`): https://github.com/Harvard-Ophthalmology-AI-Lab/FairSeg
- Harvard-FairVLMed (referred to as `fairvlmed10k`): https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP

`data/10k/` and `data/fairvlmed10k/` contain the `data_summary.csv` and
`filter_file.txt` files used in our experiments. Age below 65 is labelled
`young`, otherwise `elderly`.

Move the downloaded datasets under `data/`:

```
data
├── 10k
│   ├── All          # data_XXXXX.npz (contains slo_fundus, disc_cup_mask, ...)
│   ├── data_summary.csv
│   └── filter_file.txt
└── fairvlmed10k
    ├── All          # data_XXXXX.npz
    ├── mask         # data_XXXXX_predict.png (OC/OD masks predicted with SAM)
    ├── data_summary.csv
    └── filter_file.txt
```

Download the SD 2.1 ControlNet init checkpoint and place it under `models/`:

```bash
# from https://huggingface.co/lllyasviel/ControlNet/blob/main/models/control_sd21_ini.ckpt
wget -P models/ https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd21_ini.ckpt
```

Alternatively, initialise it from a Stable Diffusion 2.1 checkpoint with:

```bash
python tool_add_control_sd21.py <sd_v21.ckpt> models/control_sd21_ini.ckpt
```

## Training

```bash
python train_gen_sd21.py \
    --dataset_dir ./data/10k \
    --result_dir ./output/glaucodiff \
    --seed 42
```

Checkpoints are saved under `./output/glaucodiff/seed42/lightning_logs/`.
Swap `--dataset_dir ./data/fairvlmed10k` to train on FairVLMed.

## Generation

```bash
python gen_sd21.py \
    --dataset_dir ./data/fairvlmed10k \
    --ckpt_path ./output/glaucodiff/seed42/lightning_logs/version_0/checkpoints/epoch=X-step=Y.ckpt \
    --result_dir ./output/generated \
    --ddim_steps 150 200
```

For each source image the pipeline emits (i) the source itself as
`{name}_target.png`, (ii) the generated SLO with the original vCDR, and
(iii) additional generations for scaled vCDR values that cover the healthy
and glaucomatous range (`*_extra_{vCDR}_*_generate.png`). The exact prompt is
written to `{name}_prompt.txt`.

## Follow-up work

For a substantially extended framework including reward-guided fine-tuning
and released model weights, see our follow-up project **ControlGlaucoma**:
https://github.com/WANG-ZIHENG/ControlGlaucoma.

## Citation

```bibtex
@inproceedings{wang2025fairness,
  title={Fairness-Aware vCDR-Controlled Generation for Glaucoma Diagnosis},
  author={Wang, Ziheng and Yang, Shuran and Chen, Wen and Zhang, Zhen and Wang, Mengyu and Zhou, Feixiang and Tian, Yu and Wang, Meng and Zhao, Yitian and Zheng, Yalin and Meng, Yanda},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={256--266},
  year={2025},
  organization={Springer}
}
```

## Acknowledgements

The Stable Diffusion + ControlNet backbone is adapted from
[lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet).
