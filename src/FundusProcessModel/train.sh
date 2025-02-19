#!/bin/bash
#python Train.py --pretrain_model --model efficientnet-b0 --dataset 10k --epochs 100
#python Train.py --pretrain_model --model efficientnet-b0 --dataset 10k  --use_fake_data --epochs 100
#python Train.py --pretrain_model  --model efficientnet-b0  --dataset fairvlmed10k --epochs 100
#python Train.py --pretrain_model --model efficientnet-b3 --dataset 10k
#python Train.py --pretrain_model --model efficientnet-b3 --dataset 10k --use_fake_data
#python Train.py --pretrain_model --model efficientnet-b7 --dataset 10k
#python Train.py --pretrain_model --model efficientnet-b7 --dataset 10k --use_fake_data

#git checkout .
#git pull
#45160
#python Train.py --pretrain_model  --model efficientnet-b0 --dataset 10k --use_fake_data --balance_data --balance_attribute gender --epochs 100
#python Train.py --pretrain_model  --model efficientnet-b0 --dataset 10k --use_fake_data --balance_data --balance_attribute race --epochs 100



#49124
#python Train.py --pretrain_model  --model efficientnet-b0 --dataset 10k --use_fake_data --balance_data --balance_attribute maritalstatus --epochs 100
#python Train.py --pretrain_model  --model efficientnet-b0 --dataset 10k --use_fake_data --balance_data --balance_attribute age --epochs 100
#20567
#python Train.py --pretrain_model  --model efficientnet-b0 --dataset 10k --use_fake_data --balance_data --balance_attribute label --epochs 100
#python Train.py --pretrain_model  --model efficientnet-b0 --dataset 10k --use_fake_data --balance_data --balance_attribute ethnicity --epochs 100
#python Train.py --pretrain_model  --model efficientnet-b0 --dataset 10k --use_fake_data --balance_data --balance_attribute language --epochs 100
python Train.py --pretrain_model  --model efficientnet-b0 --dataset fairvlmed10k --use_fake_data  --epochs 100 --top_precentege 0
python Train.py --pretrain_model  --model efficientnet-b0 --dataset fairvlmed10k --use_fake_data  --epochs 100 --top_precentege 1
#python Train.py --pretrain_model  --model efficientnet-b0 --dataset fairvlmed10k --use_fake_data  --epochs 100 --top_precentege 0.5
shutdown


