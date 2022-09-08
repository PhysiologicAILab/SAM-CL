import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.tools.module_helper import ModuleHelper

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone):
        super(Decoder, self).__init__()
        if 'resnet' in backbone or 'drn' in backbone:
            low_level_inplanes = 256
        elif 'xception' in backbone:
            low_level_inplanes = 128
        elif 'mobilenet' in backbone:
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = ModuleHelper.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       ModuleHelper.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       ModuleHelper.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        feat_map = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(feat_map)
        return x, feat_map

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def build_decoder(num_classes, backbone):
    return Decoder(num_classes, backbone)
