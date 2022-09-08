import torch.nn as nn
import torch.nn.functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.modules.decoder_block import DeepLabHead
from lib.models.modules.projection import ProjectionHead

# from lib.models.modules.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from lib.models.tools.module_helper import ModuleHelper

from lib.models.modules.deeplab_aspp import build_aspp
from lib.models.modules.deeplab_decoder import build_decoder


class DeepLabV3Contrast(nn.Module):
    def __init__(self, configer):
        super(DeepLabV3Contrast, self).__init__()

        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        self.proj_dim = self.configer.get('contrast', 'proj_dim')

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]

        self.proj_head = ProjectionHead(dim_in=in_channels[1], proj_dim=self.proj_dim)

        self.decoder = DeepLabHead(num_classes=self.num_classes, bn_type=self.configer.get('network', 'bn_type'))

        for modules in [self.proj_head, self.decoder]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, x_, with_embed=False, is_eval=False):
        x = self.backbone(x_)

        embedding = self.proj_head(x[-1])

        x = self.decoder(x[-4:])

        return {'embed': embedding, 'seg_aux': x[1], 'seg': x[0]}


class DeepLabV3(nn.Module):

    def __init__(self, configer):
        super(DeepLabV3, self).__init__()

        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')

        self.n_channels = 1
        self.n_classes = self.configer.get('data', 'num_classes')
        self.n_features = 256+48    #304

        backbone_name = self.configer.get('network', 'backbone')
        if 'drn' in backbone_name:
            output_stride = 8
        else:
            output_stride = 16

        self.backbone = BackboneSelector(configer).get_backbone()
        self.aspp = build_aspp(backbone_name, output_stride)
        self.decoder = build_decoder(self.n_classes, backbone_name)
        self.freeze_bn = False

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x, feat_map = self.decoder(x, low_level_feat)

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        feat_map = F.interpolate(feat_map, size=input.size()[2:], mode='bilinear', align_corners=True)

        return feat_map, x

    def freeze_bn(self):
        for m in self.modules():
            m.eval()


# class DeepLabV3(nn.Module):
#     def __init__(self, configer):
#         super(DeepLabV3, self).__init__()

#         self.configer = configer
#         self.num_classes = self.configer.get('data', 'num_classes')
#         self.backbone = BackboneSelector(configer).get_backbone()

#         self.decoder = DeepLabHead(num_classes=self.num_classes, bn_type=self.configer.get('network', 'bn_type'))

#         for m in self.decoder.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight.data)
#                 if m.bias is not None:
#                     m.bias.data.zero_()

#     def forward(self, x_):
#         x = self.backbone(x_)

#         x = self.decoder(x[-4:])

#         return {'seg_aux': x[1], 'seg': x[0]}
#         # return x[1], x[0]
