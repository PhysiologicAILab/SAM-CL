import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.tools.module_helper import ModuleHelper

class Decoder(nn.Module):
    def __init__(self, configer):
        super(Decoder, self).__init__()

        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        backbone_name = self.configer.get('network', 'backbone')

        if 'resnet' in backbone_name or 'drn' in backbone_name or 'xception' in backbone_name:
            low_level_inplanes = 256
        elif 'mobilenet' in backbone_name:
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.bn_type = self.configer.get('network', 'bn_type')
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = ModuleHelper.BNReLU(48, bn_type=self.bn_type)
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       ModuleHelper.BNReLU(256, bn_type=self.bn_type),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       ModuleHelper.BNReLU(256, bn_type=self.bn_type),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        feat_map = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(feat_map)
        return x, feat_map

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            # m.weight.data.fill_(1)
            # m.bias.data.zero_()


def build_decoder(configer):
    return Decoder(configer)
