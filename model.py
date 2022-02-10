#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
import torch.nn.functional as F

'''
model for binary hdc
'''

class FastSign(torch.nn.Module):
    '''
    This is a fast version of the SignActivation.
    '''
    def __init__(self):
        super(FastSign, self).__init__()

    def forward(self, input):
        out_forward = torch.sign(input)
        out_backward = torch.clamp(input, -1.3, 1.3)
        return out_forward.detach() - out_backward.detach() + out_backward

class BinaryLinear(torch.nn.Linear):
    '''
    A fully connected layer with weights binarized to {-1, +1}.
    '''
    def __init__(self, in_features, out_features, bias=True):
        super(BinaryLinear, self).__init__(
            in_features, out_features, bias
        )
        self.binarize = FastSign()
    def forward(self, input):
        return F.linear(input, self.binarize(self.weight), self.bias)

class BModel(torch.nn.Module):
    def __init__(self, in_dim=32768, classes=10):
        super(BModel, self).__init__()
        self.in_dim = in_dim
        self.fc = BinaryLinear(self.in_dim, classes, bias=False)
    def forward(self, x):
        x = self.fc(x) * (1.0/self.in_dim**0.5)
        return x

    
# '''
# model for cyclic group hdc of different order
# '''

class FastRound(torch.nn.Module):
    '''
    This is a fast version of the round.
    '''
    def __init__(self):
        super(FastRound, self).__init__()

    def forward(self, input):
        out_forward = torch.round(input)
        out_backward = input #torch.clamp(input, -1.3, 1.3)
        return out_forward.detach() - out_backward.detach() + out_backward
    
class RoundLinear(torch.nn.Linear):
    '''
    A fully connected layer with weights rounded to closest integers 
    '''
    def __init__(self, in_features, out_features, gorder, bias=True):
        super(RoundLinear, self).__init__(
            in_features, out_features, bias
        )
        self.gorder = gorder
        self.Bias = bias
        self.round = FastRound()
        self.radius = torch.nn.Parameter(torch.ones(1)) # 1.0 
    
    def pts_map(self, x, r=1.0):
        theta = 2.0 * np.pi / (1.0*self.gorder) * x
        pts = r * torch.stack([torch.cos(theta), torch.sin(theta)], -1)
        return pts

    def GroupSim(self, input, weight):
        map_weight = self.pts_map(weight, r=self.radius)
        map_input = self.pts_map(input).unsqueeze(1)
        return torch.sum(torch.sum(map_weight * map_input, dim=-1), dim=-1)

    def forward(self, input):
        weight_q = self.weight
        if not self.training:
            weight_q = self.round(self.weight)
        sims = self.GroupSim(input, weight_q)
        if self.Bias:
            sims += self.bias
        return sims

class GModel(torch.nn.Module):
    def __init__(self, gorder, in_dim=32768, classes=10):
        super(GModel, self).__init__()
        self.in_dim = in_dim
        self.fc = RoundLinear(self.in_dim, classes, gorder, bias=False)
        
    def forward(self, x):
        x = self.fc(x) * (1.0/self.in_dim**0.5)
        return x