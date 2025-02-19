


# Fundust image classification

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
download fairvlmed10k or 10k datasets
```python

mv data/fairvlmed10k/* /src/fairvlmed10k
mv data/10k/* /src/10k
```
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

## 🚀 Run
### 第一步
使用controlnet项目训练并生成
```python
cd src/ControlNet_matting_input
download control_sd21_ini.ckpt to models
python train_gen_sd21.py --dataset_dir /root/fairvlmed10k
```
生成大量图像
```python
python gen_sd21.py --01-19-训练controlnet-seed5111原图抠图填充后做输入/lightning_logs/version_282/checkpoints/epoch=3-step=2160.ckpt
```
### 第二步
对生成图像进行测评，首先使用10k训练集训练一个分割模型
```python
cd src/TransUNet-main
python train.py --root_path /root/ --dataset 10k --vit_name R50-ViT-B_16
```
使用分割模型对生成图像和mask进行评分
```python
python test.py --is_savenii --root_path /sample/gen_image/fairvlmed10k --dataset sd_gen0 --vit_name R50-ViT-B_16
or
python test.py --is_savenii --root_path /sample/gen_image/10k --dataset sd_gen0 --vit_name R50-ViT-B_16
```
运行完分割模型部分将会获得sd_gen_metric{seed}.csv文件


### 第三步
整理生成数据形成以下结构
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
│           ├── data_00008_d3ab_extra_0.23.png
│           ├── data_00008_d3ab_extra_0.23_generate.png
│       ├── sd_gen_metric.csv
│   ├── fairvlmed10k
│       ├── sd_gen0
│           ├── data_00003_6a47_extra_0.22.png
│           ├── data_00003_6a47_extra_0.22_generate.png
│       ├── sd_gen_metric.csv
``` 

测评分类性能
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
如果仅测试模型效果，请通过以下代码执行
```python
python --model efficientnet-b0 --data_root /root/ --dataset fairvlmed10k --best_model_path checkpoints/label_classification/wt5ic8g1/best_auc.pth
```
