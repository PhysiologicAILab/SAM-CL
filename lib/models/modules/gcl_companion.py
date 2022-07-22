import torch
import torch.nn as nn
import numpy as np
import torch.nn.utils.spectral_norm as spectral_norm
from lib.models.tools.module_helper import ModuleHelper


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, apply_spectral_norm=True, bn_type='torchsyncbn'):
        super(DownConv, self).__init__()
        n_ch1 = in_channels
        n_ch2 = in_channels # in_channels // 2
        n_ch3 = out_channels #out_channels // 3

        if apply_spectral_norm:
            self.conv = nn.Sequential(
                spectral_norm(nn.Conv2d(n_ch1, n_ch2, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)),
                ModuleHelper.BNReLU(n_ch2, bn_type=bn_type),
                spectral_norm(nn.Conv2d(n_ch2, n_ch3, kernel_size=3, stride=2, padding=1, padding_mode='reflect', bias=False)),
                ModuleHelper.BNReLU(n_ch3, bn_type=bn_type),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(n_ch1, n_ch2, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
                ModuleHelper.BNReLU(n_ch2, bn_type=bn_type),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=3, stride=2, padding=1, padding_mode='reflect', bias=False),
                ModuleHelper.BNReLU(n_ch3, bn_type=bn_type),
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvFinal(nn.Module):
    def __init__(self, n_ch1, n_ch2, apply_spectral_norm=True, bn_type='torchsyncbn'):
        super(ConvFinal, self).__init__()

        if apply_spectral_norm:
            self.conv_final = nn.Sequential(
                spectral_norm(nn.Conv2d(n_ch1, n_ch2, kernel_size=1, stride=1, padding=1, padding_mode='reflect', bias=False)),
                ModuleHelper.BatchNorm2d(bn_type=bn_type)(n_ch2),
            )
        else:
            self.conv_final = nn.Sequential(
                nn.Conv2d(n_ch1, n_ch2, kernel_size=1, stride=1, padding=1, padding_mode='reflect', bias=False),
                ModuleHelper.BNReLU(n_ch2, bn_type=bn_type)(n_ch2),
            )

    def forward(self, x):
        x = self.conv_final(x)
        return x


class GCL_Companion(nn.Module):
    def __init__(self, configer):
        super(GCL_Companion, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')

        self.bn_type = self.configer.get('network', 'bn_type')
        self.apply_spectral_norm = bool(self.configer.get('gcl', 'apply_spectral_norm'))
        self.n_channels = int(self.configer.get('data', 'num_channels'))

        self.nf = self.num_classes * self.n_channels
        # self.nf = self.num_classes + self.n_channels
        self.n_filters = np.array([1*self.nf, 2*self.nf, 4*self.nf, 8*self.nf, 1])
        self.conv_down_1 = DownConv(self.n_filters[0] , self.n_filters[1], apply_spectral_norm=self.apply_spectral_norm, bn_type=self.bn_type)
        self.conv_down_2 = DownConv(self.n_filters[1], self.n_filters[2], apply_spectral_norm=self.apply_spectral_norm, bn_type=self.bn_type)
        self.conv_down_3 = DownConv(self.n_filters[2], self.n_filters[3], apply_spectral_norm=self.apply_spectral_norm, bn_type=self.bn_type)
        self.conv_final = ConvFinal(self.n_filters[3], self.n_filters[4], apply_spectral_norm=self.apply_spectral_norm, bn_type=self.bn_type)

    def forward(self, input_img, seg_map):

        b, _, h, w = input_img.shape
        # print("input_img.shape", input_img.shape)
        # print("seg_map.shape", seg_map.shape)

        if self.n_channels == 1:
            x0 = input_img * seg_map

        else:
            x0_0 = torch.mul(input_img[:, 0, :, :].expand(b, self.num_classes, h, w), seg_map)
            x0_1 = torch.mul(input_img[:, 1, :, :].expand(b, self.num_classes, h, w), seg_map)
            x0_2 = torch.mul(input_img[:, 2, :, :].expand(b, self.num_classes, h, w), seg_map)
            x0 = torch.cat([x0_0, x0_1, x0_2], dim=1)

        # x0 = torch.cat([input_img, seg_map], dim=1)

        x1 = self.conv_down_1(x0)
        x2 = self.conv_down_2(x1)
        x3 = self.conv_down_3(x2)
        x4 = self.conv_final(x3)
        return [x1, x2, x3, x4]