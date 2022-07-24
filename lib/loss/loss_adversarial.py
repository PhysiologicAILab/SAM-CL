# coding=utf-8

"""
The implementation of the paper:
Region Mutual Information Loss for Semantic Segmentation.
"""

# python 2.X, 3.X compatibility
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from abc import ABC

import torch
import torch.nn as nn
from lib.loss.pytorch_ssim import SSIM
from lib.utils.tools.logger import Logger as Log


class GCL_Loss(nn.Module, ABC):
    def __init__(self, configer=None):
        super(GCL_Loss, self).__init__()

        self.configer = configer
        loss_type = self.configer.get("gcl", "contrastive_loss")
        if loss_type.lower() == "ssim":
            self.lossObj_x1_1 = SSIM(window_size=9)
            self.lossObj_x2_1 = SSIM(window_size=7)
            self.lossObj_x3_1 = SSIM(window_size=5)
            self.lossObj_x4_1 = SSIM(window_size=3)
            self.lossObj_x1_2 = SSIM(window_size=9)
            self.lossObj_x2_2 = SSIM(window_size=7)
            self.lossObj_x3_2 = SSIM(window_size=5)
            self.lossObj_x4_2 = SSIM(window_size=3)
            self.lossObj_x1_3 = SSIM(window_size=9)
            self.lossObj_x2_3 = SSIM(window_size=7)
            self.lossObj_x3_3 = SSIM(window_size=5)
            self.lossObj_x4_3 = SSIM(window_size=3)
            self.loss_sign = 1
            Log.info('Using SSIM Loss')
        else:
            self.lossObj_x1_1 = nn.SmoothL1Loss()
            self.lossObj_x2_1 = nn.SmoothL1Loss()
            self.lossObj_x3_1 = nn.SmoothL1Loss()
            self.lossObj_x4_1 = nn.SmoothL1Loss()
            self.lossObj_x1_2 = nn.SmoothL1Loss()
            self.lossObj_x2_2 = nn.SmoothL1Loss()
            self.lossObj_x3_2 = nn.SmoothL1Loss()
            self.lossObj_x4_2 = nn.SmoothL1Loss()
            self.lossObj_x1_3 = nn.SmoothL1Loss()
            self.lossObj_x2_3 = nn.SmoothL1Loss()
            self.lossObj_x3_3 = nn.SmoothL1Loss()
            self.lossObj_x4_3 = nn.SmoothL1Loss()
            self.loss_sign = -1
            Log.info('Using L1 Loss')

        self.real_feat_sign = 1.0
        self.fake_feat_sign = -1.0

    def forward(self, critic_outputs_real, critic_outputs_fake, critic_outputs_pred=None, with_pred_seg=False, **kwargs):

        real_seg_x1, real_seg_x2, real_seg_x3, real_seg_x4 = critic_outputs_real
        fake_seg_x1, fake_seg_x2, fake_seg_x3, fake_seg_x4 = critic_outputs_fake

        if with_pred_seg:
            pred_seg_x1, pred_seg_x2, pred_seg_x3, pred_seg_x4 = critic_outputs_pred

        if with_pred_seg:
            loss = (
                (0.20) * self.lossObj_x1_1(1 + real_seg_x1, 1 + (self.real_feat_sign * self.loss_sign * pred_seg_x1)) +
                (0.20) * self.lossObj_x2_1(1 + real_seg_x2, 1 + (self.real_feat_sign * self.loss_sign * pred_seg_x2)) +
                (0.20) * self.lossObj_x3_1(1 + real_seg_x3, 1 + (self.real_feat_sign * self.loss_sign * pred_seg_x3)) +
                (0.20) * self.lossObj_x4_1(1 + real_seg_x4, 1 + (self.real_feat_sign * self.loss_sign * pred_seg_x4)) +

                (0.20) * self.lossObj_x1_2(1 + real_seg_x1, 1 + (self.fake_feat_sign * self.loss_sign * fake_seg_x1)) +
                (0.20) * self.lossObj_x2_2(1 + real_seg_x2, 1 + (self.fake_feat_sign * self.loss_sign * fake_seg_x2)) +
                (0.20) * self.lossObj_x3_2(1 + real_seg_x3, 1 + (self.fake_feat_sign * self.loss_sign * fake_seg_x3)) +
                (0.20) * self.lossObj_x4_2(1 + real_seg_x4, 1 + (self.fake_feat_sign * self.loss_sign * fake_seg_x4)) +

                (0.20) * self.lossObj_x1_3(1 + pred_seg_x1, 1 + (self.fake_feat_sign * self.loss_sign * fake_seg_x1)) +
                (0.20) * self.lossObj_x2_3(1 + pred_seg_x2, 1 + (self.fake_feat_sign * self.loss_sign * fake_seg_x2)) +
                (0.20) * self.lossObj_x3_3(1 + pred_seg_x3, 1 + (self.fake_feat_sign * self.loss_sign * fake_seg_x3)) +
                (0.20) * self.lossObj_x4_3(1 + pred_seg_x4, 1 + (self.fake_feat_sign * self.loss_sign * fake_seg_x4))
            )

        else:
            loss = (
                (0.25) * self.lossObj_x1_1(1 + real_seg_x1, 1 + (self.fake_feat_sign * self.loss_sign * fake_seg_x1)) +
                (0.25) * self.lossObj_x2_1(1 + real_seg_x2, 1 + (self.fake_feat_sign * self.loss_sign * fake_seg_x2)) +
                (0.25) * self.lossObj_x3_1(1 + real_seg_x3, 1 + (self.fake_feat_sign * self.loss_sign * fake_seg_x3)) +
                (0.25) * self.lossObj_x4_1(1 + real_seg_x4, 1 + (self.fake_feat_sign * self.loss_sign * fake_seg_x4))
            )

        return loss


class GAN_Loss(nn.Module, ABC):
    def __init__(self, configer=None):
        super(GAN_Loss, self).__init__()

        self.configer = configer
        self.loss_weight = self.configer.get('gan', 'loss_weight')
        self.lossObj = nn.BCEWithLogitsLoss()

    def forward(self, real, fake_pred, real_pred, **kwargs):

        loss = (
            (1.00) * self.lossObj(real, torch.ones_like(real)) +
            (1.00) * self.lossObj(real_pred, torch.ones_like(real_pred)) +
            (1.00) * self.lossObj(fake_pred, torch.zeros_like(fake_pred))
        )

        return self.loss_weight * loss
