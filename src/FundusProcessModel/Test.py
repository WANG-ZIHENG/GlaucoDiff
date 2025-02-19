import os
from collections import defaultdict
import numpy as np
import torch
from utils import get_datasets
from Models import Model
from torch.utils import data
from torch import nn

import logger
import utils
from utils import get_args
from misc import plot_confusion_matrix
from sklearn.metrics import classification_report
import sklearn.metrics as mtc
from sklearn.metrics import confusion_matrix
import wandb
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from misc import print_metrics, training_curve
# from openTSNE import TSNE
from sklearn.manifold import TSNE
import tsneutil
from evaluator import getAUC,getACC
import argparse

def test_net(epoch, model, test_generator,cls_num_list_train, device, criterion, args):
    # def test_net(model,test_generator,device,criterion,args):

    cls_num_list = test_generator.dataset.cls_num_list
    break_point = int(len(cls_num_list) / 4)
    many_shot_thr = cls_num_list[break_point]
    low_shot_thr = cls_num_list[-break_point]
    cls_num_list_test = test_generator.dataset.cls_num_list
    cls_num = len(test_generator.dataset.cls_num_list)
    num_steps = 0
    test_loss = 0
    correct = 0
    correct_class = [0] * cls_num
    test_metrics = defaultdict(float)
    all_labels_d = torch.tensor([], dtype=torch.long).to(device)
    all_predictions_d = torch.tensor([], dtype=torch.long).to(device)
    all_predictions_probabilities_d = torch.tensor([], dtype=torch.float).to(device)
    all_output_d = []
    all_attrs = []

    # loss_weight_alpha = 1 - (epoch/args.max_epochs)**2
    # args.CE_loss_weight, args.CCL_loss_weight = loss_weight_alpha, (1-loss_weight_alpha)

    for data in test_generator:
        images = data["image"]
        labels = data["target"]
        label_and_attributes = data["label_and_attributes"]
        all_attrs.append(label_and_attributes.cpu().numpy())
        # Transfer to GPU:
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        features, output = model(images)

        loss = criterion(output, labels)


        num_steps += bsz

        test_loss += loss.item() * bsz


        predicted_probability, predicted = torch.max(output, dim=1)
        all_output_d.append(output.cpu().detach().numpy())

        correct += (predicted == labels).sum()
        all_labels_d = torch.cat((all_labels_d, labels), 0)
        all_predictions_d = torch.cat((all_predictions_d, predicted), 0)
        all_predictions_probabilities_d = torch.cat((all_predictions_probabilities_d, predicted_probability), 0)
        for l in range(0, cls_num):
            correct_class[l] += (predicted[labels == l] == labels[labels == l]).sum()
        
    all_output_d = np.concatenate(all_output_d)
    y_true = all_labels_d.cpu()
    y_predicted = all_predictions_d.cpu()
    all_output_softmax = torch.softmax(torch.tensor(all_output_d), dim=1).numpy()
    all_attrs = np.concatenate(all_attrs, axis=0)
    
    # 这里假设y_true是类别标签的一维数组
    # 如果你的任务是二分类，确保y_true和all_output_softmax的形状是兼容的
    num_classes = len(np.unique(y_true))
    if num_classes == 2:
        y_true_one_hot = np.eye(num_classes)[y_true]  # Convert to one-hot encoding for binary classification
    else:
        y_true_one_hot = y_true

   

    
    # all_output_d = np.concatenate(all_output_d)
    # y_true = all_labels_d.cpu()
    # y_predicted = all_predictions_d.cpu()  # to('cpu')
    valset_predicted_probabilites = all_predictions_probabilities_d.cpu()  # to('cpu')
    # ����QWKָ��
    qwk_score = cohen_kappa_score(y_true, y_predicted, weights='quadratic')
    micro_precision = mtc.precision_score(y_true, y_predicted, average="micro")
    micro_recall = mtc.recall_score(y_true, y_predicted, average="micro")
    micro_f1 = mtc.f1_score(y_true, y_predicted, average="micro")

    macro_precision = mtc.precision_score(y_true, y_predicted, average="macro")
    macro_recall = mtc.recall_score(y_true, y_predicted, average="macro")
    macro_f1 = mtc.f1_score(y_true, y_predicted, average="macro")
    mcc = mtc.matthews_corrcoef(y_true, y_predicted)
    y_true = y_true.detach().numpy()
    acc = getACC(y_true, all_output_softmax, task=test_generator.dataset.task)
    if test_generator.dataset.task == 'binary-class':
        all_output_softmax = np.max(all_output_softmax, axis=1)
    auc = getAUC(y_true, all_output_softmax, task=test_generator.dataset.task)

    test_es_acc, overall_auc, test_es_auc, test_aucs_by_attrs, test_dpds, test_eods, between_group_disparity, specificity, sensitivity, f1, precision = utils.evalute_comprehensive_perf(
        all_output_softmax, y_true, all_attrs.T)

    logger.logkv('epoch', epoch)
    logger.logkv('test_acc', acc)
    logger.logkv('test_auc', overall_auc)
    logger.logkv('test_specificity',specificity)
    logger.logkv('test_sensitivity', sensitivity)
    logger.logkv('test_f1', f1)
    logger.logkv('test_precision', precision)
    for ii in range(len(test_es_acc)):
        logger.logkv(f'test_es_acc_attr{ii}', test_es_acc[ii])
    for ii in range(len(test_es_auc)):
        logger.logkv(f'test_es_auc_attr{ii}', test_es_auc[ii])
    for ii in range(len(test_aucs_by_attrs)):
        for iii in range(len(test_aucs_by_attrs[ii])):
            logger.logkv(f'test_auc_attr{ii}_group{iii}', test_aucs_by_attrs[ii][iii])

    for ii in range(len(between_group_disparity)):
        logger.logkv(f'test_auc_attr{ii}_std_group_disparity', between_group_disparity[ii][0])
        logger.logkv(f'test_auc_attr{ii}_max_group_disparity', between_group_disparity[ii][1])

    for ii in range(len(test_dpds)):
        logger.logkv(f'test_dpd_attr{ii}', test_dpds[ii])
    for ii in range(len(test_eods)):
        logger.logkv(f'test_eod_attr{ii}', test_eods[ii])

    logger.dumpkvs()

    test_metrics['acc'] = acc
    test_metrics['auc'] = auc
    test_metrics['micro_precision'] = micro_precision
    test_metrics['micro_recall'] = micro_recall
    test_metrics['micro_f1'] = micro_f1
    test_metrics['macro_precision'] = macro_precision
    test_metrics['macro_recall'] = macro_recall
    test_metrics['macro_f1'] = macro_f1
    test_metrics['mcc'] = mcc
    test_metrics['qwk'] = qwk_score
    
    
    x = all_output_d
    y = all_labels_d.cpu().numpy()
    id_to_name = test_generator.dataset.id_to_name
    id_to_name = {int(k): v for k, v in id_to_name.items()}
    # y = [id_to_name[str(i)] for i in y]
    n_components = 2
    tsne = TSNE(
        n_components=n_components,
        # init='pca',
        perplexity=50,
        n_iter=500,
        metric="euclidean",
        # callbacks=ErrorLogger(),
        n_jobs=8,
        random_state=42,
    )
    print("fit tsne")
    embedding = tsne.fit_transform(x)
    tsneutil.plot(embedding, y, colors=tsneutil.MOUSE_10X_COLORS, save_path="tsne.png",label_order = list(id_to_name.keys()))
    wandb.log({f"tsne": wandb.Image("tsne.png", caption="")},
              commit=False)
    cm = confusion_matrix(y_true, y_predicted)  # confusion matrix

    print('Accuracy of the network on the %d test images: %f %%' % (num_steps, (100.0 * correct / num_steps)))

    print(cm)

    print("taking class names to plot CM")

    class_names = test_generator.dataset.class_names # test_datasets.classes  # taking class names for plotting confusion matrix

    print("Generating confution matrix")

    plot_confusion_matrix(cm, classes=class_names, title='my confusion matrix')
    print(test_metrics)
    print_metrics(test_metrics, args.max_epochs)

    ##################################################################
    # classification report
    #################################################################
    classification_report_str = classification_report(y_true, y_predicted, target_names=class_names)
    print(classification_report_str)
    os.makedirs('classification_report', exist_ok=True)
    with open('classification_report/classification_report.txt', 'w') as f:
        f.write(classification_report_str)
    # artifact = wandb.Artifact(name="classification_report", type="dataset")
    # artifact.add_dir(local_path="classification_report")

    return (test_loss / num_steps), test_metrics, num_steps


if __name__ == '__main__':

    args = get_args()
    _, test_dataset, _ = get_datasets(args)
    test_generator = data.DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=8)

    cls_num_list = test_dataset.cls_num_list
    n_classes = len(cls_num_list)
    model = Model(args, n_classes=n_classes,
                       pretrained=args.pretrain_model)  # make weights=True if you want to download pre-trained weights


    checkpoint = torch.load(args.best_model_path, map_location=args.device)  # loading best model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    num_list = test_generator.dataset.cls_num_list
    wandb.init(
               name='Test'
               , config=args.__dict__, job_type='train', mode='offline')
    with torch.no_grad():
        test_loss, test_metrics, test_num_steps = test_net(0, model, test_generator, num_list,
                                                           args.device, criterion, args)
    print_metrics(test_metrics, test_num_steps)



