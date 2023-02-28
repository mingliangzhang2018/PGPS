import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DotProduct(nn.Module):
    def __init__(self, num_classes=1000, feat_dim=2048, bias=True):
        super(DotProduct, self).__init__()
        # print('<DotProductClassifier> contains bias: {}'.format(bias))
        self.fc = nn.Linear(feat_dim, num_classes,bias)
        
    def forward(self, x, *args):
        x = self.fc(x)
        return x


class CosNorm(nn.Module):
    def __init__(self, in_dims, out_dims, scale=16, margin=0.5, init_std=0.001):
        super(CosNorm, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.Tensor(out_dims, in_dims).cuda())
        self.reset_parameters() 

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, *args):
        norm_x = torch.norm(input.clone(), 2, 1, keepdim=True)
        ex = (norm_x / (1 + norm_x)) * (input / norm_x)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        return torch.mm(self.scale * ex, ew.t())


class FCNorm(nn.Module):
    # for LDAM Loss
    def __init__(self, num_features, num_classes, scale=20.0):
        super(FCNorm, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.scale = scale

    def forward(self, x):
        out = self.scale * F.linear(F.normalize(x), F.normalize(self.weight))
        return out


class DistFC(nn.Module):

    def __init__(self, num_features, num_classes,init_weight=True):
        super(DistFC, self).__init__()
        self.centers=nn.Parameter(torch.randn(num_features,num_classes).cuda(),requires_grad=True)
        if init_weight:
            self.__init_weight()

    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers)

    def forward(self, x):
        features_square=torch.sum(torch.pow(x,2),1, keepdim=True)
        centers_square=torch.sum(torch.pow(self.centers,2),0, keepdim=True)
        features_into_centers=2.0*torch.matmul(x, (self.centers))
        dist=features_square+centers_square-features_into_centers   
        return self.centers, dist


