import copy
import logging
import math
import torch
from torch import nn
import timm
from torch.nn import functional as F
from torchvision.models import resnet18, resnet34, resnet50
import numpy as np

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()
        self.use_RP=False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.sigma is not None:
            nn.init.constant_(self.sigma, 1)

    def forward(self, input):
        if not self.use_RP:
            out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        else:
            if self.W_rand is not None:
                inn = torch.nn.functional.relu(input @ self.W_rand)
            else:
                inn=input
                #inn=torch.bmm(input[:,0:100].unsqueeze(-1), input[:,0:100].unsqueeze(-2)).flatten(start_dim=1) #interaction terms instead of RP
            out = F.linear(inn,self.weight)

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
        )
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
            
    def forward(self, x):
        return F.relu(self.block(x) + self.shortcut(x))

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(x.size(-1))
        attention = F.softmax(scores, dim=-1)
        return torch.matmul(attention, v)

class SELayer(nn.Module):
    """Squeeze-and-Excitation层"""
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        y = self.avg_pool(x.unsqueeze(-1)).squeeze(-1)
        y = self.fc(y)
        return x * y.expand_as(x)

def get_convnet(args, pretrained=False):
    name = args["convnet_type"].lower()
    
    if name == "resnet18":
        model = nn.Sequential(
            # 特征提取层
            nn.Linear(18, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            SELayer(256),  # 添加SE注意力
            
            # 特征转换层1
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            SELayer(512),
            
            # 特征转换层2
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            SELayer(512),
            
            # 特征聚合层
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        
        # 初始化权重
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        model.out_dim = 512
        return model
    else:
        raise NotImplementedError("Unknown type {}".format(name))

class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()
        self.convnet = get_convnet(args, pretrained)
        self.fc = None
        self.feature_dim = self.convnet.out_dim

    def forward(self, x):
        features = self.convnet(x)  # 直接使用特征
        out = self.fc(features)
        return {"features": features, "logits": out["logits"]}

class ResNetCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        # 初始化时设置分类层
        self.update_fc(10)  # 初始设置10个类别
        
    def update_fc(self, nb_classes):
        fc = CosineLinear(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc
        
    def forward(self, x):
        features = self.convnet(x)  # 直接使用特征
        out = self.fc(features)
        return {"features": features, "logits": out["logits"]}

class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.update_fc(10)  # 也为SimpleVitNet添加初始化分类层

    def update_fc(self, nb_classes):
        fc = CosineLinear(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def forward(self, x):
        features = self.convnet(x)
        features = features.view(features.size(0), -1)  # 展平特征
        out = self.fc(features)
        return {"features": features, "logits": out["logits"]}
