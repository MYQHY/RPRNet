# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# @Author  : Yongqi Mu
# @File    : STDNet.py
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torch.nn as nn
import torch
import torch.nn.functional as F
from .heatmaphead import *
from .IRGSM_SAMPLE import *
from .DeformConv3D import *
class Conv3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)
    
class ResBlock3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv3d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(ch)
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)
class ECA3D(nn.Module):
    """Efficient Channel Attention for 3D features"""
    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # [B, C, 1, 1, 1]
        y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1))
        return x * y.expand_as(x)
class DeformConv3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=(3,3,3), s=(1,1,1), p=1,
                 max_disp_xy=0.0, max_disp_t=0.0):
        super().__init__()
        self.conv = DeformConv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            bias=False,
            max_disp_xy=max_disp_xy,
            max_disp_t=max_disp_t,
        )
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))   
class RPRNet(nn.Module):
    def __init__(self,in_channels,num_frame):
        super().__init__()
        base_ch=64
        
               
        self.stem = Conv3DBlock(3, base_ch//2, k=(3,3,3), s=(1,1,1), p=(1,1,1))

        # Stage1
        self.stage1 = nn.Sequential(
            DeformConv3DBlock(base_ch//2, base_ch, k=(3,3,3), s=(1,2,2), p=1,
                                  max_disp_xy=0.5, max_disp_t=0.0),
            ResBlock3D(base_ch),
            ECA3D()
        )

        # Stage2
        self.stage2 = nn.Sequential(
            DeformConv3DBlock(base_ch, base_ch*2, k=(3,3,3), s=(2,1,1), p=1,
                                  max_disp_xy=0.0, max_disp_t=0.5),
            ResBlock3D(base_ch*2),
            ECA3D()
        )

        # Stage3
        self.stage3 = nn.Sequential(
            DeformConv3DBlock(base_ch*2, base_ch*4, k=(3,3,3), s=(1,1,1), p=1,
                                  max_disp_xy=0, max_disp_t=0),
            ResBlock3D(base_ch*4),
            ECA3D()
        )
        self.pooling3d = nn.MaxPool3d(kernel_size=(3,1,1),stride=(3,1,1))
        # 输出变换
        self.out_conv = nn.Conv3d(base_ch*4, 256, kernel_size=1, bias=False)
        self.out_bn = nn.BatchNorm3d(256)
        self.out_relu = nn.ReLU(inplace=True)

    
        self.IRGSM_SAMPLE = IRGSM_SAMPLE(256)
        channelinput_seghead = 256
        self.MultiScaleSegHead = MultiScaleSegHead(in_ch=channelinput_seghead, mid_ch=256, out_ch=1, single_scale_only=False)
        

    def forward(self, x):
        """
        x: [B, C, T, H, W]
        returns: (features_flat, logits)
        """
        B, C, T, H, W = x.shape #[B,3,6,H,W]
        y = self.stem(x)              # [B,32,6,H,W]
        y = self.stage1(y)            # [B,64,6,H/2,W/2]
        y = self.stage2(y)            # [B,128,3,H/4,W/4]
        y = self.stage3(y)            # [B,256,3,H/8,W/8]
        
        y = self.pooling3d(y)           # 时间维平均 → [B,256,H/8,W/8]
        y = self.out_relu(self.out_bn(self.out_conv(y))).squeeze(2)    
        y = self.IRGSM_SAMPLE(y)
        y = self.MultiScaleSegHead(y)

        return y