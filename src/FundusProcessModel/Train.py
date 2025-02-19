#encoding=utf-8
#9998(78) bin/python3(28)
import torch
import torchvision
import time
from torch import nn
from torch import optim
from torch.utils import data
from datetime import datetime

from Models import Model
from Validate import validate_net
from Test import test_net
from misc import print_metrics, training_curve 
from PIL import Image
import os
import re
from collections import defaultdict
import numpy as np
import logging
import csv
from torchvision import transforms, datasets, models
import sklearn.metrics as mtc
import mydatasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tqdm import tqdm
from losses import LDAMLoss, LogitAdjust, SupConLoss, MixLoss, Fairness
import torch.nn.functional as F
import wandb
from utils import get_datasets,get_args
import copy
from sklearn.metrics import cohen_kappa_score,roc_auc_score
from evaluator import getAUC,getACC
from utils import adjust_lr
###########################
# Checking if GPU is used
###########################




########################################
# Setting basic parameters for the model
########################################


         

args=get_args()
batch_size=args.batch_size
max_epochs=args.max_epochs
lr=args.lr
device = args.device




####################################
# Training
####################################




def add_prefix(dct, prefix):
    return {f'{prefix}-{key}': val for key, val in dct.items()}


class train:
    def __init__(self):
        self.args = get_args()
        training_dataset, test_dataset, validation_dataset = get_datasets(self.args)

        # group_size_on_race, group_size_on_gender, group_size_on_ethnicity = count_number_of_groups(trn_havo_dataset)
        # logger.log(f'group size on race in training set: {group_size_on_race}')
        # logger.log(f'group size on gender in training set: {group_size_on_gender}')
        # logger.log(f'group size on ethnicity in training set: {group_size_on_ethnicity}')
        # group_size_on_race, group_size_on_gender, group_size_on_ethnicity = count_number_of_groups(val_havo_dataset)
        # logger.log(f'group size on race in test set: {group_size_on_race}')
        # logger.log(f'group size on gender in test set: {group_size_on_gender}')
        # logger.log(f'group size on ethnicity in test set: {group_size_on_ethnicity}')

        self.cls_num_list = training_dataset.cls_num_list
        self.n_classes = len(self.cls_num_list)
        self.training_generator=data.DataLoader(training_dataset, batch_size, shuffle=True, num_workers=8,) # ** unpacks a dictionary into keyword arguments
        self.validation_generator=data.DataLoader(validation_dataset,batch_size,shuffle=False,num_workers=8)
        self.test_generator=data.DataLoader(test_dataset,batch_size,shuffle=False,num_workers=8)

        print("Number of Each Class of Training set images:{}".format(training_dataset.cls_num_list))
        print('Number of Training set images:{}'.format(len(training_dataset)))
        print('Number of Validation set images:{}'.format(len(validation_dataset)))
        print('Number of Test set images:{}'.format(len(test_dataset)))

        # Initialize model
        self.model = Model(args, n_classes=self.n_classes,
                           pretrained=args.pretrain_model)  # make weights=True if you want to download pre-trained weights
        # model.load_state_dict(torch.load('./densenet121.pth',map_location='cuda'))   # provide a .pth path for already downloaded weights; otherwise comment this line out
        # Option to freeze model weights
        for param in self.model.parameters():
            param.requires_grad = True  # Set param.requires_grad = False if you want to train only the last updated layers and freeze all other layers

        self.model.to(device)



    def train_net(self):
        model = self.model


        optimizer=optim.Adam(model.parameters(), lr, weight_decay=1e-4)
        scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=4,verbose=True)

        # criterion = Fairness(attrs_name=['race', 'gender', 'ethnicity', 'language','maritalstatus','age'])
        criterion = nn.CrossEntropyLoss()


        val_micro_f1_max=0.0
        val_macro_f1_max=0.0
        val_acc_max=0.0
        val_auc_max=0.0
        epochs=[]
        lossesT=[]
        lossesV=[]
        cls_num_list_train = self.training_generator.dataset.cls_num_list
        for epoch in range(max_epochs):
            print('Epoch {}/{}'.format(epoch+1,max_epochs))
            print('-'*10)
            since=time.time()
            train_metrics=defaultdict(float)
            total_loss=0
            running_corrects=0
            num_steps=0
            
            all_labels_d = torch.tensor([], dtype=torch.long).to(device)
            all_predictions_d = torch.tensor([], dtype=torch.long).to(device)
            all_predictions_probabilities_d = []
            all_softmax_output = []


            model.train()


            #Training
            for data in tqdm(self.training_generator):
                #Transfer to GPU:
                images = data["image"]
                labels = data["target"]
                label_and_attributes = data["label_and_attributes"]

                adjust_lr(optimizer, epoch, args)



                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                bsz = labels.shape[0]

                # compute loss
                features,output = model(images)


                loss = criterion(output,labels)
                predicted_probability, predicted = torch.max(output, dim=1)
                softmax_output = F.softmax(output,dim=1)


                num_steps+=bsz
                
                optimizer.zero_grad()
                loss.backward()
                # 在更新权重之前，对梯度进行裁剪，使其不超过0.5
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
                optimizer.step()
                total_loss+=loss.item()*bsz



                running_corrects += torch.sum(predicted == labels.data)
                all_labels_d = torch.cat((all_labels_d, labels), 0)
                all_predictions_d = torch.cat((all_predictions_d, predicted), 0)
                all_softmax_output.append(softmax_output.cpu().detach().numpy())

                
            y_true = all_labels_d.cpu()
            y_predicted = all_predictions_d.cpu()  # to('cpu')
            all_softmax_output = np.concatenate(all_softmax_output)
            
            #############################
            # Standard metrics 
            #############################
            # 计算QWK指标
            train_qwk_score = cohen_kappa_score(y_true, y_predicted, weights='quadratic')
            train_micro_precision=mtc.precision_score(y_true, y_predicted, average="micro")     
            train_micro_recall=mtc.recall_score(y_true, y_predicted, average="micro")
            train_micro_f1=mtc.f1_score(y_true, y_predicted, average="micro")  
        
            train_macro_precision=mtc.precision_score(y_true, y_predicted, average="macro")     
            train_macro_recall=mtc.recall_score(y_true, y_predicted, average="macro")
            train_macro_f1=mtc.f1_score(y_true, y_predicted, average="macro")  
        
            train_mcc=mtc.matthews_corrcoef(y_true, y_predicted)

            y_true = y_true.detach().numpy()
            acc = getACC(y_true, all_softmax_output, task=self.training_generator.dataset.task)
            if self.training_generator.dataset.task == 'binary-class':
                all_softmax_output = np.max(all_softmax_output, axis=1)
            auc = getAUC(y_true, all_softmax_output, task=self.training_generator.dataset.task)
            train_metrics["lr"] = optimizer.param_groups[-1]['lr']
            train_metrics['acc'] =acc
            train_metrics['auc'] =auc
            train_metrics['loss']=total_loss/num_steps




        
            train_metrics['micro_precision']=train_micro_precision
            train_metrics['micro_recall']=train_micro_recall
            train_metrics['micro_f1']=train_micro_f1
            train_metrics['macro_precision']=train_macro_precision
            train_metrics['macro_recall']=train_macro_recall
            train_metrics['macro_f1']=train_macro_f1
            train_metrics['mcc']=train_mcc
            train_metrics['qwk'] = train_qwk_score
            
            print('Training...')
            print('Train_loss:{:.3f}'.format(total_loss/num_steps))
           
            
            print_metrics(train_metrics,num_steps)
            wandb.log(add_prefix(train_metrics, f'train'), step=epoch, commit=False)

            ############################
            # Validation
            ############################

            model.eval()
            with torch.no_grad():
                val_loss, val_metrics, val_num_steps=validate_net(epoch,model,self.validation_generator,cls_num_list_train,device,criterion,args)
                #val_loss, val_metrics, val_num_steps=validate_net(model,self.validation_generator,device,criterion,args)
                
            # scheduler.step(val_loss)
            epochs.append(epoch)
            lossesT.append(total_loss/num_steps)
            lossesV.append(val_loss)
            
            print('.'*5)
            print('Validating...')
            print('val_loss:{:.3f}'.format(val_loss))
        
            print_metrics(val_metrics,val_num_steps)


            ##################################################################
            # Writing epoch-wise training and validation results to a csv file 
            ##################################################################

            key_name=['Epoch','Train_loss','Train_micro_precision','Train_micro_recall','Train_micro_f1','Train_macro_precision','Train_macro_recall','Train_macro_f1','Train_mcc','Val_loss','Val_micro_precision','Val_micro_recall','Val_micro_f1','Val_macro_precision','Val_macro_recall','Val_macro_f1','Val_mcc']
            train_list=[]
            train_list.append(epoch)

            try:
                with open(filename+str('.csv'), 'a',newline="") as f:
                    wr = csv.writer(f,delimiter=",")
                    if epoch==0:
                        wr.writerow(key_name)

                    for k, vl in train_metrics.items():
                        train_list.append(vl)

                    train_list.append(val_loss)

                    for k, vl in val_metrics.items():
                        train_list.append(vl)
                    zip(train_list)
                    wr.writerow(train_list)


            except IOError:
                print("I/O Error")

            
            ##############################
            # Saving best model 
            ##############################
            
            if val_metrics['micro_f1']>=val_micro_f1_max:
                print('val micro f1 increased ({:.6f}-->{:.6f})'.format(val_micro_f1_max,val_metrics['micro_f1']))
                best_micro_model_path = model_path + f'/best_micro.pth'
                torch.save({'epoch':epoch+1,
                            'model_state_dict':model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'loss':val_loss},best_micro_model_path)

               
                val_micro_f1_max=val_metrics['micro_f1']

            if val_metrics['macro_f1'] >= val_macro_f1_max:
                print('val macro f1 increased ({:.6f}-->{:.6f})'.format(val_macro_f1_max,
                                                                                     val_metrics['macro_f1']))
                best_macro_model_path = model_path + f'/best_macro.pth'
                torch.save({'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'loss': val_loss}, best_macro_model_path)


                val_macro_f1_max = val_metrics['macro_f1']

            if val_metrics['acc'] >= val_acc_max:
                print('val acc increased ({:.6f}-->{:.6f})'.format(val_acc_max,
                                                                                     val_metrics['acc']))
                best_acc_model_path = model_path + f'/best_acc.pth'
                torch.save({'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'loss': val_loss}, best_acc_model_path)
                val_acc_max = val_metrics['acc']

            if val_metrics['auc'] >= val_auc_max:
                print('val auc increased ({:.6f}-->{:.6f})'.format(val_auc_max,
                                                                                     val_metrics['auc']))
                best_auc_model_path = model_path + f'/best_auc.pth'
                torch.save({'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'loss': val_loss}, best_auc_model_path)


                val_auc_max = val_metrics['auc']

            val_metrics['val_macro_f1_max'] = val_macro_f1_max
            val_metrics['val_micro_f1_max'] = val_micro_f1_max
            val_metrics['val_auc_max'] = val_auc_max
            val_metrics['val_acc_max'] = val_acc_max
            wandb.log(add_prefix(val_metrics, f'val'), step=epoch, commit=True)
            print('-'*10)

        time_elapsed=time.time()-since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
        training_curve(epochs,lossesT,lossesV)
        epochs.clear()
        lossesT.clear()
        lossesV.clear()
        best_model_paths = {"macro":best_macro_model_path,"micro":best_micro_model_path,'acc':best_acc_model_path ,'auc':best_auc_model_path}
        test_metrics_dict = {"macro":None, "micro":None,'acc':None,'auc':None}
        ############################
        #         Test
        ############################
        for name,best_model_path in best_model_paths.items():
            test_list=[]
            best_model=self.model
            checkpoint=torch.load(best_model_path,map_location=device)   # loading best model
            best_model.load_state_dict(checkpoint['model_state_dict'])
            best_model.to(device)
            best_model.eval()
            print(f'选取best {name} model')
            with torch.no_grad():
                   test_loss, test_metrics, test_num_steps=test_net(epoch,best_model,self.test_generator,cls_num_list_train,device,criterion,args)
            print_metrics(test_metrics,test_num_steps)
            test_list.append(test_loss)


            for k, vl in test_metrics.items():
                test_list.append(vl)              # append metrics results in a list



            ##################################################################
            # Writing test results to a csv file
            ##################################################################

            key_name=['Test_loss','Test_micro_precision','Test_micro_recall','Test_micro_f1','Test_macro_precision','Test_macro_recall','Test_macro_f1','Test_mcc']
            try:

                    with open(filename+str('.csv'), 'a',newline="") as f:
                        wr = csv.writer(f,delimiter=",")
                        wr.writerow(key_name)
                        zip(test_list)
                        wr.writerow(test_list)
                        wr.writerow("")
            except IOError:
                    print("I/O Error")
            test_metrics_dict[name] = test_metrics
        return val_metrics, test_metrics_dict
        
        
                       
         
                
if __name__=="__main__":



    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device: {device}')
    logging.info(f'''Starting training:
                 Epochs: {max_epochs}
                 Batch Size: {batch_size}
                 Learning Rate: {lr}''')





    wandb_name = f"{args.model}_PreTrain_{args.pretrain_model}_{args.dataset}_"
    if args.use_fake_data:
        wandb_name += f'use{args.top_precentege}fake_data_'
    if args.balance_data:
        wandb_name += f'{args.balance_attribute}balance_'



    if os.path.exists("/D_share"):
        mode = "offline"
    else:
        mode = "online"

    wandb.init(dir=os.path.abspath("wandb"), project=f"Fundus_{args.model_task}",entity='orange_jam',
               name=wandb_name
               , config=args.__dict__, job_type='train', mode=mode)

    wandb.run.log_code(".", include_fn=lambda path: path.endswith('.py')
                                                    or path.endswith('.yaml')
                                                    or path.endswith('.sh')
                                                    or path.endswith('.txt')
                                                    or path.endswith('.md') or  path.endswith('.MD')
                                                    or path.endswith('.MD')
                       )
    # 获取当前日期
    now = datetime.now()

    # 生成简单日期字符串，例如 0931
    simple_date = now.strftime("%m%d")
    model_path = os.path.join('checkpoints',
                              args.model_task,f"{simple_date}{wandb.run.id}")  # set path to the folder that will store model's checkpoints
    os.makedirs(model_path, exist_ok=True)

    global val_micro_f1_max
    global val_macro_f1_max

    try:
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
    except OSError as err:
        print(err)

    print("Directory '% s' created" % model_path)
    filename = 'results_e' + str(max_epochs) + '_' + 'b' + str(batch_size) + '_' + 'lr' + str(
        lr) + '_' + args.model  # filename used for saving epoch-wise training details and test results

    t=train()
    t.train_net()
  




