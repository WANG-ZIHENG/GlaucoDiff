import torch
import torchvision
import time
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,args,n_classes, pretrained=False):
        super().__init__()
        device = args.device
        if args.model == "DenseNet121":
            model = torchvision.models.densenet121(weights=pretrained).to(device)
        elif args.model == "efficientnet-b0":
            model = torchvision.models.efficientnet_b0(pretrained= pretrained).to(device)
        elif args.model == "efficientnet-b3":
            model = torchvision.models.efficientnet_b3(pretrained= pretrained).to(device)
        elif args.model == "efficientnet-b7":
            model = torchvision.models.efficientnet_b7(pretrained= pretrained).to(device)
        elif args.model == "resnet18":
            model = torchvision.models.resnet18(weights=pretrained).to(device)
        elif args.model == "resnet34":
            model = torchvision.models.resnet34(weights=pretrained).to(device)
        elif args.model == "resnet50":
            model = torchvision.models.resnet50(weights=pretrained).to(device)
        elif args.model == "resnext50_32x4d":
            model = torchvision.models.resnext50_32x4d(weights=pretrained).to(device)
        elif args.model == 'resnet32':
            from models.resnet import resnet32
            assert pretrained == False
            model = resnet32(pretrained=pretrained,

                             phase_train=False,
                             norm_out=False,
                             add_rsg=False,
                             head_lists=[0,1],
                             add_arc_margin_loss=False,
                             add_add_margin_loss=False,
                             add_sphere_loss=False,
                             epoch_thresh=100,
                             )
            model = model.to(device)
        elif args.model == 'resnet10':
            from models.resnet import resnet10
            assert pretrained == False
            model = resnet10(pretrained=pretrained,

                             phase_train=False,
                             norm_out=False,
                             add_rsg=False,
                             head_lists=[0,1],
                             add_arc_margin_loss=False,
                             add_add_margin_loss=False,
                             add_sphere_loss=False,
                             epoch_thresh=100,
                             )
            model = model.to(device)

        self.features = nn.ModuleList(model.children())[:-1]
        self.features = nn.Sequential(*self.features)
        if args.model == "efficientnet-b0":
            n_inputs = nn.ModuleList(model.children())[-1][1].in_features
        else:
            n_inputs = nn.ModuleList(model.children())[-1].in_features

        self.classifier = nn.Sequential(
            nn.Linear(n_inputs, n_classes),
        )
        self.args = args
    def forward(self, input_imgs):
        features = self.features(input_imgs)
        fea = F.relu(features, inplace=True)
        fea = F.adaptive_avg_pool2d(fea, (1, 1))
        fea = torch.flatten(fea, 1)
        ce_output = self.classifier(fea)
        return features,ce_output
