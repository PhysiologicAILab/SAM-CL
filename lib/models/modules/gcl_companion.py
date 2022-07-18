import torch
import torch.nn as nn
import numpy as np
import torch.nn.utils.spectral_norm as spectral_norm
from lib.models.tools.module_helper import ModuleHelper


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, bn_type = 'torchsyncbn'):
        super(DownConv, self).__init__()
        n_ch1 = in_channels
        n_ch2 = in_channels # in_channels // 2
        n_ch3 = out_channels #out_channels // 3

        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(n_ch1, n_ch2, kernel_size=3, stride=1, padding=1)),
            ModuleHelper.BNReLU(n_ch2, bn_type=bn_type),
            spectral_norm(nn.Conv2d(n_ch2, n_ch3, kernel_size=3, stride=2, padding=1)),
            ModuleHelper.BNReLU(n_ch3, bn_type=bn_type),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvFinal(nn.Module):
    def __init__(self, n_ch1, n_ch2, bn_type='torchsyncbn'):
        super(ConvFinal, self).__init__()

        self.conv_final = nn.Sequential(
            spectral_norm(nn.Conv2d(n_ch1, n_ch2, kernel_size=1, stride=1, padding=1)),
            ModuleHelper.BNReLU(n_ch2, bn_type=bn_type),
        )

    def forward(self, x):
        x = self.conv_final(x)
        return x


class GCL_Critic(nn.Module):
    def __init__(self, configer):
        super(GCL_Critic, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')

        self.n_channels = int(self.configer.get('data', 'num_channels'))
        self.batch_size = int(self.configer.get('train', 'batch_size'))

        self.nf = self.num_classes * self.n_channels
        self.n_filters = np.array([1*self.nf, 2*self.nf, 4*self.nf, 8*self.nf, self.num_classes])
        self.conv_down_1 = DownConv(self.n_filters[0] , self.n_filters[1])
        self.conv_down_2 = DownConv(self.n_filters[1], self.n_filters[2])
        self.conv_down_3 = DownConv(self.n_filters[2], self.n_filters[3])
        self.conv_final = ConvFinal(self.n_filters[3], self.n_filters[4])

    def forward(self, input_img, seg_map):

        # if self.n_channels == 1:

        # else:
        #     x0_0 = input_img[:, 0, :, :] * seg_map
        #     x0_1 = input_img[:, 1, :, :] * seg_map
        #     x0_2 = input_img[:, 2, :, :] * seg_map
        #     x0 = torch.cat([x0_0, x0_1, x0_2], dim=1)

        x0 = input_img * seg_map
        x1 = self.conv_down_1(x0)
        x2 = self.conv_down_2(x1)
        x3 = self.conv_down_3(x2)
        x4 = self.conv_final(x3)
        return [x0, x1, x2, x3, x4]