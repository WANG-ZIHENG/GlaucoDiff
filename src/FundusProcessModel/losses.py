#encoding=utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from cvxopt import matrix, spdiag, solvers
import torchmetrics
class LogitAdjust(nn.Module):

    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)
class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class CCL(nn.Module):
    # Center Contrast Loss
    def __init__(self, margin=1, alpha=0.5):
        super(CCL, self).__init__()
        self.margin = margin
        self.alpha = alpha

    def forward(self, features, labels, class_centers):
        # intra class loss
        centers_batch = torch.stack([class_centers[label.item()] for label in labels])
        distance = torch.zeros(centers_batch.shape[0]).to(features.device)
        for i in range(features.shape[1]):
            distance += (features[:,i,:] - centers_batch).pow(2).sum(1)
        distance = distance/2
        intra_loss = distance.mean()

        # inter class loss
        # dict to list for visiting class centers via index
        centers_list = list(class_centers.values())

        inter_loss = 0
        for i in range(len(centers_list)):
            for j in range(i + 1, len(centers_list)):
                center_dist = (centers_list[i] - centers_list[j]).pow(2).sum()
                inter_loss += 1/center_dist

        inter_loss = (inter_loss / (len(centers_list) * (len(centers_list) - 1) / 2))
        # total loss
        loss = (1 - self.alpha) * intra_loss + self.alpha * inter_loss

        # l2_reg = torch.tensor(0., device=features.device)
        # for param in self.parameters():
        #     l2_reg += torch.norm(param, p=2)

        # lambda_l2 = 0.01  # Regularization strength
        # loss = loss + lambda_l2 * l2_reg


        return loss, distance



class Fairness(nn.Module):

    def __init__(self,attrs_name):
        super(Fairness, self).__init__()
        self.ce = nn.CrossEntropyLoss()

        accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)

        # metric_mean = {i:torchmetrics.Mean() for i in attrs_name}








    def forward(self, features, labels):
        ce_loss = self.ce(features, labels)






        return ce_loss



class BaseLoss(nn.Module):
    def __init__(self, num_class_list,device,):
        super(BaseLoss, self).__init__()
        self.num_class_list = np.array(num_class_list)
        self.no_of_class = len(self.num_class_list)
        self.device = device


        self.class_weight_power = 1.2
        self.class_extra_weight = np.array([1.0]*len(num_class_list))
        self.scheduler = 'cls' #'cls'
        self.drw_epoch = 50
        self.cls_epoch_min = 20
        self.cls_epoch_max = 60
        self.weight = None
    def reset_epoch(self,epoch):
        if self.scheduler == "cls":  # cumulative learning strategy
            if epoch <= self.cls_epoch_min:
                now_power = 0
            elif epoch < self.cls_epoch_max:
                now_power = ((epoch - self.cls_epoch_min) / (self.cls_epoch_max - self.cls_epoch_min)) ** 2
                now_power = now_power * self.class_weight_power
            else:
                now_power = self.class_weight_power

            per_cls_weights = 1.0 / (self.num_class_list.astype(np.float64))
            per_cls_weights = per_cls_weights * self.class_extra_weight
            per_cls_weights = [math.pow(num, now_power) for num in per_cls_weights]
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * self.no_of_class
        # print("class weight of loss: {}".format(per_cls_weights))
        self.weight = torch.FloatTensor(per_cls_weights).to(self.device)
class FocalLoss(BaseLoss):
    def __init__(self,cls_num_list,device,gamma=2,type='sigmoid',sigmoid='enlarge'):
        super(FocalLoss, self).__init__(cls_num_list,device)
        self.gamma = gamma
        self.type = type
        self.sigmoid = sigmoid
        if self.type == "ldam":
            max_m = None
            m_list = 1.0 / np.sqrt(np.sqrt(self.num_class_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.FloatTensor(m_list).to(self.device)
            self.m_list = m_list
            self.s = 30

    def forward(self, x, target):
        labels_one_hot = F.one_hot(target, self.no_of_class).float().to(self.device)
        weights = self.weight
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)

        if self.type == "sigmoid":
            weights = weights.unsqueeze(1)
            weights = weights.repeat(1, self.no_of_class)

            loss = F.binary_cross_entropy_with_logits(input=x, target=labels_one_hot, reduction="none")
            if self.gamma == 0.0:
                modulator = 1.0
            else:
                modulator = torch.exp(-self.gamma * labels_one_hot * x
                                      - self.gamma * torch.log(1 + torch.exp(-1.0 * x)))

            loss = modulator * loss
            weighted_loss = weights * loss
            if self.sigmoid == "enlarge":
                weighted_loss = torch.mean(weighted_loss) * 30
            else:
                weighted_loss = weighted_loss.sum() / weights.sum()
        elif self.type == "cross_entropy":
            loss = F.cross_entropy(x, target, reduction='none')

            p = torch.exp(-loss)
            loss = (1 - p) ** self.gamma * loss
            weighted_loss = weights * loss
            weighted_loss = weighted_loss.sum() / weights.sum()
        elif self.type == "ldam":
            index = torch.zeros_like(x, dtype=torch.uint8)
            index.scatter_(1, target.data.view(-1, 1), 1)

            index_float = index.type(torch.FloatTensor)
            index_float = index_float.to(self.device)
            batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
            batch_m = batch_m.view((-1, 1))
            x_m = x - batch_m

            output = torch.where(index, x_m, x)
            loss = F.cross_entropy(self.s * output, target, reduction='none')

            p = torch.exp(-loss)
            loss = (1 - p) ** self.gamma * loss
            weighted_loss = weights * loss
            weighted_loss = weighted_loss.sum() / weights.sum()
        else:
            raise AttributeError(
                "focal loss type can only be 'sigmoid', 'cross_entropy' and 'ldam'.")

        return weighted_loss
class LOWLoss(BaseLoss):
    """
    LOW loss, Details of the theorem can be viewed in the paper
       "LOW: Training Deep Neural Networks by Learning Optimal Sample Weights"

    Args:
        lamb (float): higher lamb means more smoothness -> weights closer to 1
    """
    def __init__(self,cls_num_list,device):
        super(LOWLoss, self).__init__(cls_num_list,device)

        self.lamb = 0.1         #https://github.com/cajosantiago/LOW/blob/master/main.py
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='none')  # can replace this with any loss with "reduction='none'"

    def forward(self, x, target):
        if x.requires_grad:         # for train
            # Compute loss gradient norm
            output_d = x.detach()
            loss_d = torch.mean(self.loss_func(output_d.requires_grad_(True), target), dim=0)
            loss_d.backward(torch.ones_like(loss_d))
            loss_grad = torch.norm(output_d.grad, 2, 1)

            # Computed weighted loss
            low_weights = self.compute_weights(loss_grad, self.lamb)
            loss = self.loss_func(x, target)
            low_loss = loss * low_weights
        else:                       # for valid
            low_loss = self.loss_func(x, target)

        labels_one_hot = F.one_hot(target, self.no_of_class).float().to(self.device)
        weights = self.weight
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)

        weighted_loss = weights * low_loss
        weighted_loss = weighted_loss.sum() / weights.sum()
        return weighted_loss

    def compute_weights(self, loss_grad, lamb):
        device = loss_grad.get_device()
        loss_grad = loss_grad.data.cpu().numpy()

        # Compute Optimal sample Weights
        aux = -(loss_grad ** 2 + lamb)
        sz = len(loss_grad)
        P = 2 * matrix(lamb * np.identity(sz))
        q = matrix(aux.astype(np.double))
        A = spdiag(matrix(-1.0, (1, sz)))
        b = matrix(0.0, (sz, 1))
        Aeq = matrix(1.0, (1, sz))
        beq = matrix(1.0 * sz)
        solvers.options['show_progress'] = False
        solvers.options['maxiters'] = 20
        solvers.options['abstol'] = 1e-4
        solvers.options['reltol'] = 1e-4
        solvers.options['feastol'] = 1e-4
        sol = solvers.qp(P, q, A, b, Aeq, beq)
        w = np.array(sol['x'])

        return torch.squeeze(torch.tensor(w, dtype=torch.float, device=device))

class GHMCLoss(BaseLoss):
    def __init__(self,cls_num_list,device,):

        super(GHMCLoss, self).__init__(cls_num_list,device)
        self.bins = 10
        self.momentum = 0.0
        self.edges = torch.arange(self.bins + 1).float().to(self.device) / self.bins
        self.edges[-1] += 1e-6
        if self.momentum > 0:
            self.acc_sum = torch.zeros(self.bins).to(self.device)
    def forward(self, pred, target):
        labels_one_hot = F.one_hot(target, self.no_of_class).float().to(self.device)
        weights = self.weight
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.no_of_class)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - labels_one_hot)

        tot = pred.numel()
        ghm_weights = torch.zeros_like(pred)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= self.edges[i]) & (g < self.edges[i+1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if self.momentum > 0:
                    self.acc_sum[i] = self.momentum * self.acc_sum[i] \
                        + (1 - self.momentum) * num_in_bin
                    ghm_weights[inds] = tot / self.acc_sum[i]
                else:
                    ghm_weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            ghm_weights = ghm_weights / n

        loss = F.binary_cross_entropy_with_logits(pred, labels_one_hot, reduction="none")
        ghm_loss = loss * ghm_weights

        weighted_loss = weights * ghm_loss
        weighted_loss = weighted_loss.sum() / weights.sum()
        return weighted_loss
class CCELoss(BaseLoss):
    """
    CCE loss, Details of the theorem can be viewed in the papers
       "Imbalanced Image Classification with Complement Cross Entropy" and
       "Complement objective training"
       https://github.com/henry8527/COT
    """
    def __init__(self, cls_num_list,device,):
        super(CCELoss, self).__init__( cls_num_list,device)

    def forward(self, pred, target):
        labels_one_hot = F.one_hot(target, self.no_of_class).float().to(self.device)
        labels_zero_hot = 1 - labels_one_hot
        weights = self.weight
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)

        pred = F.softmax(pred, dim=1)
        y_g = torch.gather(pred, 1, torch.unsqueeze(target, 1))
        y_g_no = (1 - y_g) + 1e-7           # avoiding numerical issues (first)
        p_x = pred / y_g_no.view(len(pred), 1)
        p_x_log = torch.log(p_x + 1e-10)    # avoiding numerical issues (second)
        cce_loss = p_x * p_x_log * labels_zero_hot
        cce_loss = cce_loss.sum(1)
        cce_loss = cce_loss / (self.no_of_class - 1)

        ce_loss = F.cross_entropy(pred, target, reduction='none')
        weighted_loss = (ce_loss + cce_loss) * weights
        weighted_loss = weighted_loss.sum() / weights.sum()
        return weighted_loss
class MWNLoss(BaseLoss):
    """
    Multi Weighted New loss
    Args:
        gamma (float): the hyper-parameter of focal loss
        beta (float, 0.0 - 0.4):
        type: "zero", "fix", "decrease"
        sigmoid: "normal", "enlarge"
    """
    def __init__(self,  cls_num_list,device):
        super(MWNLoss, self).__init__( cls_num_list,device)

        self.gamma = 2
        self.beta = 0.1
        self.type = 'fix'
        self.sigmoid = 'enlarge'
        if self.beta > 0.4 or self.beta < 0.0:
            raise AttributeError(
                "For MWNLoss, the value of beta must be between 0.0 and 0.0 .")

    def forward(self, x, target):
        labels_one_hot = F.one_hot(target, self.no_of_class).float().to(self.device)
        weights = self.weight
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.no_of_class)

        loss = F.binary_cross_entropy_with_logits(input=x, target=labels_one_hot, reduction="none")

        if self.beta > 0.0:
            th = - math.log(self.beta)
            if self.type == "zero":
                other = torch.zeros(loss.shape).to(self.device)
                loss = torch.where(loss <= th, loss, other)
            elif self.type == "fix":
                other = torch.ones(loss.shape).to(self.device)
                other = other * th
                loss = torch.where(loss <= th, loss, other)
            elif self.type == "decrease":
                pt = torch.exp(-1.0 * loss)
                loss = torch.where(loss <= th, loss, pt * th / self.beta)
        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * labels_one_hot * x
                                  - self.gamma * torch.log(1 + torch.exp(-1.0 * x)))

        loss = modulator * loss

        weighted_loss = weights * loss
        if self.sigmoid == "enlarge":
            weighted_loss = torch.mean(weighted_loss) * 30
        else:
            weighted_loss = weighted_loss.sum() / weights.sum()
        return weighted_loss

class MixLoss(nn.Module):
    def __init__(self,cls_num_list,args):
        super(MixLoss, self).__init__()
        self.cls_num_list =cls_num_list
        self.args = args
        self.ldam_criterion = LDAMLoss(cls_num_list=self.cls_num_list)
        self.logit_criterion = LogitAdjust(cls_num_list=self.cls_num_list)
        self.scl_criterion = SupConLoss()
        self.ce_criterion = nn.CrossEntropyLoss()
        self.ccl_criterion = CCL()
        self.wce_criterion = nn.CrossEntropyLoss(reduction='none')
        self.facal_loss_criterion = FocalLoss(cls_num_list=cls_num_list,device = args.device)
        self.LOW_loss_criterion = LOWLoss(cls_num_list=cls_num_list,device = args.device)
        self.GHMC_loss_criterion = GHMCLoss(cls_num_list=cls_num_list,device = args.device)
        self.CCE_loss_criterion = CCELoss(cls_num_list=cls_num_list,device = args.device)
        self.MWN_loss_criterion = MWNLoss(cls_num_list=cls_num_list,device = args.device)



    def forward(self, features,outputs,ce_output, labels, fea_pair_output,class_centers,epoch):

        loss = 0
        ldam_loss = torch.tensor(-1)
        logit_loss = torch.tensor(-1)
        ce_loss =torch.tensor(-1)
        supcon_loss =torch.tensor(-1)
        ccl_loss = torch.tensor(-1)
        wce_loss = torch.tensor(-1)
        similarity_loss = torch.tensor(-1)
        uncertain_loss = torch.tensor(-1)
        focal_loss = torch.tensor(-1)
        LOW_loss = torch.tensor(-1)
        GHMC_loss = torch.tensor(-1)
        CCE_loss = torch.tensor(-1)
        MWN_loss = torch.tensor(-1)
        ResLT_loss = torch.tensor(-1)
        if self.args.Focal_Loss_use:
            self.facal_loss_criterion.reset_epoch(epoch)
            focal_loss = self.facal_loss_criterion(ce_output,labels)
            loss += focal_loss
        if self.args.LOW_Loss_use:
            self.LOW_loss_criterion.reset_epoch(epoch)
            LOW_loss = self.LOW_loss_criterion(ce_output,labels)
            loss += LOW_loss
        if self.args.GHMC_Loss_use:
            self.GHMC_loss_criterion.reset_epoch(epoch)
            GHMC_loss = self.GHMC_loss_criterion(ce_output,labels)
            loss += GHMC_loss
        if self.args.CCE_Loss_use:
            self.CCE_loss_criterion.reset_epoch(epoch)
            CCE_loss = self.CCE_loss_criterion(ce_output,labels)
            loss += CCE_loss
        if self.args.MWN_Loss_use:
            self.MWN_loss_criterion.reset_epoch(epoch)
            MWN_loss = self.MWN_loss_criterion(ce_output,labels)
            loss += MWN_loss


        if self.args.LDAM_loss_use:
            ldam_loss = self.ldam_criterion(outputs, labels) * self.args.LDAM_loss_weight
            loss += ldam_loss
        if self.args.Logit_loss_use:
            logit_loss = self.logit_criterion(outputs,labels) * self.args.Logit_loss_weight
            loss +=logit_loss
        if self.args.CE_loss_use:
            ce_loss = self.ce_criterion(ce_output,labels) * self.args.CE_loss_weight
            loss += ce_loss
        if self.args.CCL_loss_use:
            non_weight_ccl_loss, distances = self.ccl_criterion(fea_pair_output, labels, class_centers)
            ccl_loss = non_weight_ccl_loss * self.args.CCL_loss_weight
            loss += ccl_loss

        if self.args.supcon_loss_use:
            supcon_loss = self.scl_criterion(features,labels) * self.args.supcon_loss_weight
            loss += supcon_loss

        if self.args.WCE_loss_use:
            wce_loss = self.weighted_cross_entropy_loss(ce_output,distances, labels, self.wce_criterion)
            loss += wce_loss


        # l2_reg = torch.tensor(0., device=features.device)
        # for param in self.parameters():
        #     l2_reg += torch.norm(param, p=2)
        # lambda_l2 = 0.01
        # loss = loss + lambda_l2 * l2_reg

        return loss,ldam_loss,logit_loss,ce_loss,supcon_loss, ccl_loss,wce_loss,similarity_loss,uncertain_loss

    def weighted_cross_entropy_loss(self, ce_output, distances, labels, ce_loss):
        # 归一化距离
        normalized_distance = (distances - distances.min()) / (distances.max() - distances.min() + 1e-6)

        # 使用指数函数计算权重，以获得更平滑的权重分布
        weights = torch.exp(-normalized_distance)
        weighted_losses = ce_loss(ce_output, labels) * weights

        return weighted_losses.mean()

        return weighted_losses.mean()*1e-5
