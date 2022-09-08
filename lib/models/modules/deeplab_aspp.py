import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.tools.module_helper import ModuleHelper

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, bn_type):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = ModuleHelper.BNReLU(planes, bn_type=bn_type)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            # m.weight.data.fill_(1)
            # m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, configer):
        super(ASPP, self).__init__()

        self.configer = configer
        backbone_name = self.configer.get('network', 'backbone')
        self.bn_type = self.configer.get('network', 'bn_type')

        if 'drn' in backbone_name:
            output_stride = 8
        else:
            output_stride = 16

        if 'drn' in backbone_name:
            inplanes = 512
        elif 'mobilenet' in backbone_name:
            inplanes = 320
        else:
            inplanes = 2048

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], bn_type=self.bn_type)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], bn_type=self.bn_type)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], bn_type=self.bn_type)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], bn_type=self.bn_type)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             ModuleHelper.BNReLU(256, bn_type=self.bn_type))
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = ModuleHelper.BNReLU(256, bn_type=self.bn_type)
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            # m.weight.data.fill_(1)
            # m.bias.data.zero_()


def build_aspp(configer):
    return ASPP(configer)