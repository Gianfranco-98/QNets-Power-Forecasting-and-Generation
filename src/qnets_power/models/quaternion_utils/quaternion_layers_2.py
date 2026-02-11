"""https://github.com/onyekpeu/Quarternion-Gated-Recurrent-Unit/blob/main/QGRU.py"""

import numpy                   as np
from   numpy.random            import RandomState
import torch
from   torch.autograd           import Variable
import torch.nn.functional      as F
import torch.nn                 as nn
from   torch.nn.parameter       import Parameter
from   torch.nn                 import Module
from   qnets_power.models.quaternion_utils.quaternion_ops_2 import *
import math
import sys


class QuaternionLinearAutograd(Module):


    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='glorot', weight_init='quaternion',
                 seed=None):

        super(QuaternionLinearAutograd, self).__init__()
        self.in_features = in_features
        self.input_dim= in_features
        self.out_features = out_features
        self.out = out_features//3

        self.r_weight, self.i_weight, self.j_weight, self.k_weight = Parameter(torch.Tensor(self.in_features, self.out_features)), Parameter(torch.Tensor(self.in_features, self.out_features)), Parameter(torch.Tensor(self.in_features, self.out_features)), Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i2=self.out*2
        self.i3=self.out*3

        if bias is True:
            self.bias = Parameter(torch.Tensor(self.out_features*3))
        else:
            self.register_parameter('bias', None)
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else 1337
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {'quaternion': quaternion_init,
                 'unitary': unitary_init}[self.weight_init]
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.r_weight, self.i_weight, self.j_weight, self.k_weight, winit,
                    self.rng, self.init_criterion)

    def forward(self, input,drop):

        
        if self.bias is True:
            self.bias1 = self.bias[:self.out]
            self.bias2 = self.bias[self.out:self.i2]
            self.bias3 = self.bias[self.i2:self.i3]
        
        self.ri_weight = self.r_weight[:,:self.out]
        self.rf_weight = self.r_weight[:,self.out:self.i2]
        self.ra_weight = self.r_weight[:,self.i2:self.i3]

        self.ii_weight = self.i_weight[:,:self.out]
        self.if_weight = self.i_weight[:,self.out:self.i2]
        self.ia_weight = self.i_weight[:,self.i2:self.i3]

        
        self.ji_weight = self.j_weight[:,:self.out]
        self.jf_weight = self.j_weight[:,self.out:self.i2]
        self.ja_weight = self.j_weight[:,self.i2:self.i3]

        
        self.ki_weight = self.k_weight[:,:self.out]
        self.kf_weight = self.k_weight[:,self.out:self.i2]
        self.ka_weight = self.k_weight[:,self.i2:self.i3]

        
        cat_kernels_4_r1 = torch.cat([self.ri_weight,  self.ii_weight, self.ji_weight,  self.ki_weight], dim=0)
        cat_kernels_4_i1 = torch.cat([self.ii_weight,  self.ri_weight, -self.ki_weight, self.ji_weight], dim=0)
        cat_kernels_4_j1 = torch.cat([self.ji_weight,  self.ki_weight, self.ri_weight, -self.ii_weight], dim=0)
        cat_kernels_4_k1 = torch.cat([self.ki_weight,  -self.ji_weight, self.ii_weight, self.ri_weight], dim=0)
        
        cat_kernels_4_r2 = torch.cat([self.rf_weight,  self.if_weight, self.jf_weight,  self.kf_weight], dim=0)
        cat_kernels_4_i2 = torch.cat([self.if_weight,  self.rf_weight, -self.kf_weight, self.jf_weight], dim=0)
        cat_kernels_4_j2 = torch.cat([self.jf_weight,  self.kf_weight, self.rf_weight, -self.if_weight], dim=0)
        cat_kernels_4_k2 = torch.cat([self.kf_weight,  -self.jf_weight, self.if_weight, self.rf_weight], dim=0)
        
        cat_kernels_4_r3 = torch.cat([self.ra_weight,  self.ia_weight, self.ja_weight,  self.ka_weight], dim=0)
        cat_kernels_4_i3 = torch.cat([self.ia_weight,  self.ra_weight, -self.ka_weight, self.ja_weight], dim=0)
        cat_kernels_4_j3 = torch.cat([self.ja_weight,  self.ka_weight, self.ra_weight, -self.ia_weight], dim=0)
        cat_kernels_4_k3 = torch.cat([self.ka_weight,  -self.ja_weight, self.ia_weight, self.ra_weight], dim=0)
        
     
#        wxf, wxi, wxo, wxa
        wi = torch.cat([cat_kernels_4_r1, cat_kernels_4_i1, cat_kernels_4_j1, cat_kernels_4_k1], dim=1)
        
        wf = torch.cat([cat_kernels_4_r2, cat_kernels_4_i2, cat_kernels_4_j2, cat_kernels_4_k2], dim=1)
        
        wa = torch.cat([cat_kernels_4_r3, cat_kernels_4_i3, cat_kernels_4_j3, cat_kernels_4_k3], dim=1)
        if self.bias is True:
            output1 = torch.mm(input, wi)+self.bias1
            output2 = torch.mm(input, wf)+self.bias2
            output3 = torch.mm(input, wa)+self.bias3

        else:
            output1 = torch.mm(input, wi)
            output2 = torch.mm(input, wf)
            output3 = torch.mm(input, wa)
        return output1, output2, output3#, output4