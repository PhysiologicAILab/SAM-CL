from lib.models.tools.module_helper import ModuleHelper
import torch.nn as nn
import numpy as np
import torch.nn.utils.spectral_norm as spectral_norm


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, apply_spectral_norm = False, bn_type = 'torchsyncbn'):
        super(DownConv, self).__init__()
        n_ch1 = in_channels
        n_ch2 = in_channels # in_channels // 2
        n_ch3 = out_channels #out_channels // 3

        if apply_spectral_norm:
            self.conv = nn.Sequential(
                spectral_norm(nn.Conv2d(n_ch1, n_ch2, kernel_size=3, stride=1, padding=1)),
                ModuleHelper.BNReLU(n_ch2, bn_type=bn_type),
                spectral_norm(nn.Conv2d(n_ch2, n_ch3, kernel_size=3, stride=2, padding=1)),
                ModuleHelper.BNReLU(n_ch3, bn_type=bn_type),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(n_ch1, n_ch2, kernel_size=3, stride=1, padding=1),
                ModuleHelper.BNReLU(n_ch2, bn_type=bn_type),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=3, stride=2, padding=1),
                ModuleHelper.BNReLU(n_ch3, bn_type=bn_type),
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvFinal(nn.Module):
    def __init__(self, n_ch1, n_ch2, apply_spectral_norm=False, bn_type='torchsyncbn'):
        super(ConvFinal, self).__init__()

        if apply_spectral_norm:
            self.conv_final = nn.Sequential(
                spectral_norm(nn.Conv2d(n_ch1, n_ch2, kernel_size=1, stride=1, padding=1)),
                ModuleHelper.BNReLU(n_ch2, bn_type=bn_type),
            )
        else:
            self.conv_final = nn.Sequential(
                nn.Conv2d(n_ch1, n_ch2, kernel_size=1, stride=1, padding=1),
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
        if self.configer.exists('train_trans', 'random_crop'):
            img_dim = self.configer.get('train_trans', 'random_crop')['crop_size']
        else:
            img_dim = self.configer.get('train', 'data_transformer')['input_size']
        
        self.apply_spectral_norm = bool(self.configer.get('gcl', 'apply_spectral_norm'))
        self.n_channels = int(self.configer.get('data', 'num_channels'))
        self.batch_size = int(self.configer.get('train', 'batch_size'))

        self.nf = self.num_classes * self.n_channels
        self.n_filters = np.array([1*self.nf, 2*self.nf, 4*self.nf, 8*self.nf, self.num_classes])
        self.conv_down_1 = DownConv(self.n_filters[0] , self.n_filters[1], apply_spectral_norm=self.apply_spectral_norm)
        self.conv_down_2 = DownConv(self.n_filters[1], self.n_filters[2], apply_spectral_norm=self.apply_spectral_norm)
        self.conv_down_3 = DownConv(self.n_filters[2], self.n_filters[3], apply_spectral_norm=self.apply_spectral_norm)
        self.conv_final = ConvFinal(self.n_filters[3], self.n_filters[4], apply_spectral_norm=self.apply_spectral_norm)
        width, height = img_dim
        # self.x0 = torch.zeros(self.batch_size, self.nf, height, width)

    def forward(self, input_img, seg_map):
        # cnt = 0
        # for cls in range(self.num_classes):
        #     for ch in range(self.n_channels):
        #         self.x0[:, cnt, :, :] = input_img[:, ch, :, :] * seg_map[:, cls, :, :]
        #         cnt += 1
        # x1 = self.conv_down_1(self.x0)

        print("input_img.shape", input_img.shape)
        print("seg_map.shape", seg_map.shape)
        # exit()

        x0 = input_img * seg_map
        print("x0.shape", x0.shape)

        x1 = self.conv_down_1(x0)
        print("x1.shape", x1.shape)

        x2 = self.conv_down_2(x1)
        x3 = self.conv_down_3(x2)
        x4 = self.conv_final(x3)
        return [x0, x1, x2, x3, x4]


class GCL_Models(object):

    def __init__(self, configer):
        self.configer = configer

    def gcl_critic_model(self, **kwargs):
        model = GCL_Critic(self.configer)
        model = ModuleHelper.load_model(model)
        return model
