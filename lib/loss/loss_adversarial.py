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

class GCL_Loss(nn.Module, ABC):
    def __init__(self, configer=None):
        super(GCL_Loss, self).__init__()

        self.configer = configer

        self.lossObj_x1 = nn.SmoothL1Loss()
        self.lossObj_x2 = nn.SmoothL1Loss()
        self.lossObj_x3 = nn.SmoothL1Loss()
        self.lossObj_x4 = nn.BCEWithLogitsLoss()

    def forward(self, critic_outputs_real, critic_outputs_fake, critic_outputs_pred, with_pred_seg=False, with_fake_seg=False, **kwargs):

        real_seg_x1, real_seg_x2, real_seg_x3, real_seg_x4 = critic_outputs_real

        if with_fake_seg:
            fake_seg_x1, fake_seg_x2, fake_seg_x3, fake_seg_x4 = critic_outputs_fake

        if with_pred_seg:
            pred_seg_x1, pred_seg_x2, pred_seg_x3, pred_seg_x4 = critic_outputs_pred

        if with_fake_seg and  with_pred_seg:

            loss = (
                (0.20) * self.lossObj_x4(real_seg_x4, torch.ones_like(real_seg_x4)) +
                (0.20) * self.lossObj_x4(fake_seg_x4, torch.zeros_like(fake_seg_x4)) +
                (0.20) * self.lossObj_x4(pred_seg_x4, torch.ones_like(pred_seg_x4)) +
                (0.20) * self.lossObj_x2(real_seg_x1, pred_seg_x1) +
                (0.20) * self.lossObj_x3(real_seg_x2, pred_seg_x2) +
                (0.20) * self.lossObj_x3(real_seg_x3, pred_seg_x3) +
                (- 0.20) * self.lossObj_x3(real_seg_x3, fake_seg_x3)
            )

        elif not with_fake_seg and with_pred_seg:

            loss = (
                (0.20) * self.lossObj_x4(real_seg_x4, torch.ones_like(real_seg_x4)) +
                (0.20) * self.lossObj_x4(pred_seg_x4, torch.ones_like(pred_seg_x4)) +
                (0.20) * self.lossObj_x2(real_seg_x1, pred_seg_x1) +
                (0.20) * self.lossObj_x3(real_seg_x2, pred_seg_x2) +
                (0.20) * self.lossObj_x3(real_seg_x3, pred_seg_x3)
            )

        elif not with_pred_seg and with_fake_seg:
            loss = (
                (0.50) * self.lossObj_x4(real_seg_x4, torch.ones_like(real_seg_x4)) +
                (0.50) * self.lossObj_x4(fake_seg_x4, torch.zeros_like(fake_seg_x4))
            )

        else:
            print('Invalid loss computation, consider breaking the training')
        
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
