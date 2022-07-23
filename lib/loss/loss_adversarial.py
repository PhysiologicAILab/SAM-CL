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


class GCL_Loss_Critic(nn.Module, ABC):
    def __init__(self, configer=None):
        super(GCL_Loss_Critic, self).__init__()

        self.configer = configer
        self.lossObj = nn.SmoothL1Loss()

    def forward(self, critic_outputs_real, critic_outputs_fake, critic_outputs_pred=None, with_pred_seg=False, **kwargs):

        real_seg_x1, real_seg_x2, real_seg_x3, real_seg_x4 = critic_outputs_real
        fake_seg_x1, fake_seg_x2, fake_seg_x3, fake_seg_x4 = critic_outputs_fake

        if with_pred_seg:
            pred_seg_x1, pred_seg_x2, pred_seg_x3, pred_seg_x4 = critic_outputs_pred

        if with_pred_seg:
            loss = (
                (0.20) * self.lossObj(real_seg_x1, pred_seg_x1) +
                (0.20) * self.lossObj(real_seg_x2, pred_seg_x2) +
                (0.20) * self.lossObj(real_seg_x3, pred_seg_x3) +
                (0.20) * self.lossObj(real_seg_x4, pred_seg_x4) +

                (0.20) * self.lossObj(real_seg_x1, (1 - fake_seg_x1)) +
                (0.20) * self.lossObj(real_seg_x2, (1 - fake_seg_x2)) +
                (0.20) * self.lossObj(real_seg_x3, (1 - fake_seg_x3)) +
                (0.20) * self.lossObj(real_seg_x4, (1 - fake_seg_x4)) +

                (0.20) * self.lossObj(pred_seg_x1, (1 - fake_seg_x1)) +
                (0.20) * self.lossObj(pred_seg_x2, (1 - fake_seg_x2)) +
                (0.20) * self.lossObj(pred_seg_x3, (1 - fake_seg_x3)) +
                (0.20) * self.lossObj(pred_seg_x4, (1 - fake_seg_x4))
            )

        else:
            loss = (
                (0.25) * self.lossObj(real_seg_x1, (1 - fake_seg_x1)) +
                (0.25) * self.lossObj(real_seg_x2, (1 - fake_seg_x2)) +
                (0.25) * self.lossObj(real_seg_x3, (1 - fake_seg_x3)) +
                (0.25) * self.lossObj(real_seg_x4, (1 - fake_seg_x4))
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
