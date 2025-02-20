


# GlaucoDiff

[![GitHub license](https://img.shields.io/github/license/用户名/仓库名)](https://github.com/用户名/仓库名/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## 🛠️ Install dependencies

```python
conda create -n GlaucoDiff python=3.11.7
conda activate GlaucoDiff
pip install -r requirements.txt
```

## 📁 Data preparation

The Harvard-FairSeg dataset (named as **10k**) can be accessed via this [link](https://github.com/Harvard-Ophthalmology-AI-Lab/FairSeg?tab=readme-ov-file), and the Harvard-FairVLMed dataset (named as **fairvlmed10k**) can be accessed via this [link](https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP).

We provide the `data_summary.csv` and `filter_file.txt` files for both datasets, which contain the filenames used in our experiments, along with information on whether each file is used for the training, validation, or test set. In addition to the demographic information and medical records provided by the source data, we categorize ages under 65 as "young" and ages 65 and above as "elderly".



Move data files to your own directory path:

```python
mv data/fairvlmed10k/* /src/fairvlmed10k
mv data/10k/* /src/10k
```

Download the datasets 10k and fairvlmed10k to GlaucoDiff/data. The directory should look like

```
10k
.
|-- Test 
│   ├── data_08001.npz
│   ├── data_08002.npz
|-- Training 
│   ├── data_00001.npz
│   ├── data_00002.npz
|-- Validation 
│   ├── data_07515.npz
│   ├── data_07516.npz
|-- data_summary.csv
|-- filter_file.txt
```

```
fairvlmed10k
.
|-- Test 
│   ├── data_08001.npz
│   ├── data_08002.npz
|-- Training 
│   ├── data_00001.npz
│   ├── data_00002.npz
|-- Validation 
│   ├── data_07001.npz
│   ├── data_07002.npz
|-- data_summary.csv
|-- filter_file.txt
```

## 🚀 Train

### First

**The complete code for generating glaucoma images will be made publicly available after the paper is accepted.**



### Second

**Here, we provide only the training code and pre-trained models for TransUNet. The code for SAM, Unet, and the complete sample selection process will be made publicly available after the paper is published.**

Trained segmentation model:

```python
cd src/TransUNet-main
python train.py --root_path /root/ --dataset 10k --vit_name R50-ViT-B_16
```

Download pre-trained segmentation model and place it in the specified directory:

| Model Name                                     | Download Link                                                | Description                                                  |
| ---------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224 | [Google Drive](https://drive.google.com/drive/folders/1WwHKwmoVH73ENMX6rrhaEmGW6TVFpCQK?usp=sharing) | Place the pre-trained segmentation model under the directory `scr/TransUNet-main/model` |

Use segmentation models to grade the generated images.

```python
python test.py --is_savenii --root_path /sample/gen_image/10k --dataset sd_gen0 --vit_name R50-ViT-B_16
or
python test.py --is_savenii --root_path /sample/gen_image/fairvlmed10k --dataset sd_gen0 --vit_name R50-ViT-B_16
```

After running the segmentation model, you will obtain the file `sd_gen_metric{seed}.csv`


### Third

Organize the generated data to form the following structure:

```
data_root
.
|-- 10k 
│   ├── Test
│   ├── Training
│   ├── Validation
│   ├── data_summary.csv
│   ├── filter_file.txt
|-- fairvlmed10k 
│   ├── Test
│   ├── Training
│   ├── Validation
│   ├── data_summary.csv
│   ├── filter_file.txt
|-- gen_image 
│   ├── 10k
│       ├── sd_gen0
│           ├── data_00008_d3ab.png
│           ├── data_00008_d3ab_generate.png
│       ├── sd_gen_metric.csv
│   ├── fairvlmed10k
│       ├── sd_gen0
│           ├── data_00003_6a47.png
│           ├── data_00003_6a47_generate.png
│       ├── sd_gen_metric.csv
```



### Forth

Training the classification model

```python
cd src/FundusProcessModel
python Train.py --pretrain_model  --model efficientnet-b0 --data_root /root/ --dataset fairvlmed10k --use_fake_data  --epochs 100 --top_precentege 0
```

Common parameters:

- --model: Model selection
- --epochs: Number of epochs
- --device: Training device (cuda:0/cpu)
- --data_root: Root directory of the dataset
- --dataset: Name of the dataset used
- --pretrain_model: Whether to use a pre-trained model
- --use_fake_data: Whether to use generated data
- --top_percentage: Percentage of generated data to use
- --balance_data: Whether to balance the quantities of each class
- --balance_attribute: Attribute to balance
- --best_model_path: Path to the loaded model (useful only for running the test set)

## 🧪 only Test

Download pre-trained models and place it in the specified directory:

| Model                                 | Download Link                                                | Description                                                  |
| ------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| efficientnet-b0 10k_best_auc          | [Google Drive](https://drive.google.com/drive/folders/1NDjwGxQ4oiQm5Lvt-MvY-ar2ZZcMu8OU?usp=sharing) | Trained by the 10k training set with generated samples       |
| efficientnet-b0 fairvlmed10k_best_auc | [Google Drive](https://drive.google.com/drive/folders/1s3HoRg4pwJcS1TA4wiwl8s8NXewfcSRF?usp=sharing) | Trained by the fairvlmed10k training set with generated samples |

If only testing the model performance, please run the code below:


```python
python --model efficientnet-b0 --data_root /root/ --dataset fairvlmed10k --best_model_path checkpoints/label_classification/wt5ic8g1/best_auc.pth
```
