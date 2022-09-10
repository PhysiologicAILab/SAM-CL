from abc import ABC
import torch
import torch.nn as nn
# from lib.utils.tools.logger import Logger as Log
# from lib.loss.dice_loss import DiceLoss
from lib.loss.rmi_loss import RMILoss
# from lib.loss.loss_helper import FSCELoss

class SAMCL_Loss_4(nn.Module, ABC):
    def __init__(self, configer=None):
        super(SAMCL_Loss_4, self).__init__()

        self.configer = configer
        # self.num_classes = self.configer.get('data', 'num_classes')
        # class_mode = 'multilabel'
        # classes = list(range(self.num_classes))
        # log_loss = True #False #True

        # self.ce_loss = FSCELoss(self.configer)

        # weight = None
        # if self.configer.exists('loss', 'params') and 'ce_weight' in self.configer.get('loss', 'params'):
        #     weight = self.configer.get('loss', 'params')['ce_weight']
        #     weight = torch.FloatTensor(weight).cuda()

        # reduction = 'mean'
        # if self.configer.exists('loss', 'params') and 'ce_reduction' in self.configer.get('loss', 'params'):
        #     reduction = self.configer.get('loss', 'params')['ce_reduction']

        # ignore_index = -1
        # if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
        #     ignore_index = self.configer.get('loss', 'params')[
        #         'ce_ignore_index']

        # self.lossObj_x0 = nn.TripletMarginWithDistanceLoss(distance_function = DiceLoss(self.configer))
        self.lossObj_x0 = nn.TripletMarginWithDistanceLoss(distance_function = RMILoss(self.configer))
        # self.lossObj_x0 = nn.TripletMarginWithDistanceLoss(distance_function = nn.CrossEntropyLoss())
        # self.lossObj_x0 = nn.TripletMarginWithDistanceLoss(distance_function = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction))
        self.lossObj_x1 = nn.TripletMarginWithDistanceLoss(distance_function = nn.CrossEntropyLoss())
        self.lossObj_x2 = nn.TripletMarginWithDistanceLoss(distance_function = nn.CrossEntropyLoss())
        self.lossObj_x3 = nn.TripletMarginWithDistanceLoss(distance_function = nn.CrossEntropyLoss())

        # self.real_feat_sign = 1.0
        # self.fake_feat_sign = -1.0
        # self.weight_full = 1.0
        # self.weight_half = 0.5

    def forward(self, critic_outputs_real, critic_outputs_fake, critic_outputs_pred, **kwargs):

        real_seg_x0, real_seg_x1, real_seg_x2, real_seg_x3 = critic_outputs_real
        fake_seg_x0, fake_seg_x1, fake_seg_x2, fake_seg_x3 = critic_outputs_fake
        pred_seg_x0, pred_seg_x1, pred_seg_x2, pred_seg_x3 = critic_outputs_pred

        real_seg_x0_label_enc = torch.argmax(real_seg_x0, dim=1)
        fake_seg_x0_label_enc = torch.argmax(fake_seg_x0, dim=1)

        loss = (
            self.lossObj_x0(pred_seg_x0, real_seg_x0_label_enc, fake_seg_x0_label_enc) +
            self.lossObj_x1(pred_seg_x1, real_seg_x1, fake_seg_x1) +
            self.lossObj_x2(pred_seg_x2, real_seg_x2, fake_seg_x2) +
            self.lossObj_x3(pred_seg_x3, real_seg_x3, fake_seg_x3)
        )

        return loss


class SAMCL_Loss_2(nn.Module, ABC):
    def __init__(self, configer=None):
        super(SAMCL_Loss_2, self).__init__()

        self.configer = configer
        self.lossObj_x0 = nn.TripletMarginWithDistanceLoss(distance_function = RMILoss(self.configer))
        self.lossObj_x1 = nn.TripletMarginWithDistanceLoss(distance_function = RMILoss(self.configer))

    def forward(self, critic_outputs_real, critic_outputs_fake, critic_outputs_pred, **kwargs):

        real_seg_x0, real_seg_x1 = critic_outputs_real
        fake_seg_x0, fake_seg_x1 = critic_outputs_fake
        pred_seg_x0, pred_seg_x1 = critic_outputs_pred

        real_seg_x0_label_enc = torch.argmax(real_seg_x0, dim=1)
        fake_seg_x0_label_enc = torch.argmax(fake_seg_x0, dim=1)

        loss = (
            self.lossObj_x0(pred_seg_x0, real_seg_x0_label_enc, fake_seg_x0_label_enc) +
            self.lossObj_x1(pred_seg_x1, real_seg_x1, fake_seg_x1)
        )

        return loss