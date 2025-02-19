from collections import defaultdict
import numpy as np
import torch
import csv
import sklearn.metrics as mtc
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score, roc_auc_score
import torch.nn.functional as F
import utils
from evaluator import getAUC,getACC
import logger

def validate_net(epoch, model, validation_generator,cls_num_list_train, device, criterion, args):

    cls_num_list = validation_generator.dataset.cls_num_list
    break_point = int(len(cls_num_list) / 4)
    many_shot_thr = cls_num_list[break_point]
    low_shot_thr = cls_num_list[-break_point]

    cls_num_list_test = validation_generator.dataset.cls_num_list
    cls_num = len(validation_generator.dataset.cls_num_list)
    # def validate_net(model,validation_generator,device,criterion,args):
    num_steps = 0
    val_loss = 0
    correct_class = [0] * cls_num
    correct = 0
    val_metrics = defaultdict(float)
    all_labels_d = torch.tensor([], dtype=torch.long).to(device)
    all_predictions_d = torch.tensor([], dtype=torch.long).to(device)
    all_predictions_probabilities_d = torch.tensor([], dtype=torch.float).to(device)
    all_softmax_output = []
    all_attrs = []

    for data in validation_generator:
        # Transfer to GPU:

        images = data["image"]
        labels = data["target"]
        label_and_attributes = data["label_and_attributes"]
        all_attrs.append(label_and_attributes.cpu().numpy())

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        features, output = model(images)

        loss = criterion( output,labels)


        num_steps += bsz

        val_loss += loss.item() * bsz



        predicted_probability, predicted = torch.max(output, dim=1)
        softmax_output = F.softmax(output, dim=1)

        for l in range(0, cls_num):
            correct_class[l] += (predicted[labels == l] == labels[labels == l]).sum()

        correct += (predicted == labels).sum()
        all_labels_d = torch.cat((all_labels_d, labels), 0)
        all_predictions_d = torch.cat((all_predictions_d, predicted), 0)
        # all_predictions_probabilities_d = torch.cat((all_predictions_probabilities_d, predicted_probability), 0)
        all_softmax_output.append(softmax_output.cpu().detach().numpy())

    y_true = all_labels_d.cpu()
    y_predicted = all_predictions_d.cpu()  # to('cpu')
    # valset_predicted_probabilites = all_predictions_probabilities_d.cpu()  # to('cpu')
    all_softmax_output = np.concatenate(all_softmax_output)
    all_attrs = np.concatenate(all_attrs, axis=0)

    #############################
    # Standard metrics
    #############################
    qwk_score = cohen_kappa_score(y_true, y_predicted, weights='quadratic')
    micro_precision = mtc.precision_score(y_true, y_predicted, average="micro")
    micro_recall = mtc.recall_score(y_true, y_predicted, average="micro")
    micro_f1 = mtc.f1_score(y_true, y_predicted, average="micro")

    macro_precision = mtc.precision_score(y_true, y_predicted, average="macro")
    macro_recall = mtc.recall_score(y_true, y_predicted, average="macro")
    macro_f1 = mtc.f1_score(y_true, y_predicted, average="macro")

    mcc = mtc.matthews_corrcoef(y_true, y_predicted)

    y_true = y_true.detach().numpy()
    acc = getACC(y_true, all_softmax_output, task=validation_generator.dataset.task)
    if validation_generator.dataset.task == 'binary-class':
        all_softmax_output = np.max(all_softmax_output, axis=1)
    auc = getAUC(y_true, all_softmax_output, task=validation_generator.dataset.task)


    eval_es_acc, overall_auc, eval_es_auc, eval_aucs_by_attrs, eval_dpds, eval_eods, between_group_disparity, specificity, sensitivity, f1, precision = utils.evalute_comprehensive_perf(
        all_softmax_output, y_true, all_attrs.T)

    logger.logkv('epoch', epoch)
    logger.logkv('eval_acc', acc)
    logger.logkv('eval_auc', overall_auc)
    logger.logkv('eval_specificity', specificity)
    logger.logkv('eval_sensitivity', sensitivity)
    logger.logkv('eval_f1', f1)
    logger.logkv('eval_precision', precision)
    for ii in range(len(eval_es_acc)):
        logger.logkv(f'eval_es_acc_attr{ii}',eval_es_acc[ii])
    for ii in range(len(eval_es_auc)):
        logger.logkv(f'eval_es_auc_attr{ii}', eval_es_auc[ii])
    for ii in range(len(eval_aucs_by_attrs)):
        for iii in range(len(eval_aucs_by_attrs[ii])):
            logger.logkv(f'eval_auc_attr{ii}_group{iii}', eval_aucs_by_attrs[ii][iii])

    for ii in range(len(between_group_disparity)):
        logger.logkv(f'eval_auc_attr{ii}_std_group_disparity', between_group_disparity[ii][0])
        logger.logkv(f'eval_auc_attr{ii}_max_group_disparity', between_group_disparity[ii][1])

    for ii in range(len(eval_dpds)):
        logger.logkv(f'eval_dpd_attr{ii}', eval_dpds[ii] )
    for ii in range(len(eval_eods)):
        logger.logkv(f'eval_eod_attr{ii}', eval_eods[ii] )

    logger.dumpkvs()
    val_metrics['acc'] = acc

    val_metrics['auc'] = auc
    val_metrics['loss'] = val_loss / num_steps



    val_metrics['micro_precision'] = micro_precision
    val_metrics['micro_recall'] = micro_recall
    val_metrics['micro_f1'] = micro_f1
    val_metrics['macro_precision'] = macro_precision
    val_metrics['macro_recall'] = macro_recall
    val_metrics['macro_f1'] = macro_f1
    val_metrics['mcc'] = mcc
    val_metrics['qwk'] = qwk_score

    return (val_loss / num_steps), val_metrics, num_steps
