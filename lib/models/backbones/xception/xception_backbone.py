import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
# from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from lib.models.tools.module_helper import ModuleHelper


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, bn_type='torchsyncbn'):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.bn = ModuleHelper.BatchNorm2d(bn_type=bn_type)(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, bn_type='torchsyncbn',
                 start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, bn_type=bn_type))
            rep.append(ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation, bn_type=bn_type))
            rep.append(ModuleHelper.BatchNorm2d(bn_type=bn_type)(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, bn_type=bn_type))
            rep.append(ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes))

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 2, bn_type=bn_type))
            rep.append(ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes))

        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 1, bn_type=bn_type))
            rep.append(ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip

        return x


class AlignedXception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, in_channels, output_stride, bn_type='torchsyncbn'):
        super(AlignedXception, self).__init__()

        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError


        # Entry flow
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(64)

        self.block1 = Block(64, 128, reps=2, stride=2, bn_type=bn_type, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, bn_type=bn_type, start_with_relu=False,
                            grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, bn_type=bn_type,
                            start_with_relu=True, grow_first=True, is_last=True)

        # Middle flow
        self.block4  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             bn_type=bn_type, start_with_relu=True, grow_first=True)
        self.block5  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             bn_type=bn_type, start_with_relu=True, grow_first=True)
        self.block6  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             bn_type=bn_type, start_with_relu=True, grow_first=True)
        self.block7  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             bn_type=bn_type, start_with_relu=True, grow_first=True)
        self.block8  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             bn_type=bn_type, start_with_relu=True, grow_first=True)
        self.block9  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             bn_type=bn_type, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             bn_type=bn_type, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             bn_type=bn_type, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             bn_type=bn_type, start_with_relu=True, grow_first=True)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             bn_type=bn_type, start_with_relu=True, grow_first=True)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             bn_type=bn_type, start_with_relu=True, grow_first=True)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             bn_type=bn_type, start_with_relu=True, grow_first=True)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             bn_type=bn_type, start_with_relu=True, grow_first=True)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             bn_type=bn_type, start_with_relu=True, grow_first=True)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             bn_type=bn_type, start_with_relu=True, grow_first=True)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             bn_type=bn_type, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=exit_block_dilations[0],
                             bn_type=bn_type, start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=exit_block_dilations[1], bn_type=bn_type)
        self.bn3 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(1536)

        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1], bn_type=bn_type)
        self.bn4 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(1536)

        self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=exit_block_dilations[1], bn_type=bn_type)
        self.bn5 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(2048)

        # Init weights
        self._init_weight(bn_type)

        self.num_features = 2048

    def get_num_features(self):
        return self.num_features


    def forward(self, x):
        tuple_features = list()
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        tuple_features.append(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        tuple_features.append(x)

        x = self.block1(x)
        x = self.relu(x)
        x = self.block2(x)
        tuple_features.append(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        tuple_features.append(x)
        
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        tuple_features.append(x)

        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        
        # Exit flow
        x = self.block20(x)
        tuple_features.append(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        tuple_features.append(x)

        return tuple_features

    def _init_weight(self, bn_type):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, ModuleHelper.BatchNorm2d(bn_type=bn_type, ret_cls=True)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class XceptionBackbone(object):
    def __init__(self, configer):
        self.configer = configer

    def __call__(self):
        arch = self.configer.get('network', 'backbone')
        resume = self.configer.get('network', 'resume')
        in_channels = self.configer.get('data', 'num_channels')

        if arch == 'xception_16':
            arch_net = AlignedXception(in_channels=in_channels, output_stride=16, bn_type='torchsyncbn')
            if resume is None:
                arch_net = ModuleHelper.load_model(arch_net,
                                                   pretrained=self.configer.get('network', 'pretrained'),
                                                   all_match=False,
                                                   network='xception')
        elif arch == 'xception_8':
            arch_net = AlignedXception(in_channels=in_channels, output_stride=8, bn_type='torchsyncbn')
            if resume is None:
                arch_net = ModuleHelper.load_model(arch_net,
                                                   pretrained=self.configer.get('network', 'pretrained'),
                                                   all_match=False,
                                                   network='xception')

        else:
            raise Exception('Architecture undefined!')

        return arch_net


if __name__ == "__main__":
    import torch
    # model = AlignedXception(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=16)
    model = AlignedXception(1, output_stride=16, bn_type='torchsyncbn')
    input = torch.rand(1, 1, 512, 512)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
