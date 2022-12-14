#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Jitesh Joshi(jitesh.joshi.20@ucl.ac.uk)


import argparse
import os
import pdb

import numpy as np
from copy import deepcopy

from lib.utils.helpers.image_helper import ImageHelper
from lib.utils.tools.logger import Logger as Log
from lib.utils.tools.configer import Configer
from lib.metrics.running_score import RunningScore
# import matplotlib.pyplot as plt


class ThermalFaceDBEvaluator(object):
    def __init__(self, configer):
        self.configer = configer
        self.seg_running_score = RunningScore(configer)

    def relabel(self, labelmap):
        if self.configer.exists('data', 'remap_classes'):
            labelmap = self._remap_classes(labelmap, self.configer.get('data', 'remap_classes'))

        return labelmap

    def _remap_classes(self, labelmap, remap_classes):
        max_cls_val = np.max(labelmap)
        remapped_labelmap = deepcopy(labelmap)
        for cls_val in range(max_cls_val+1):
            remapped_labelmap[labelmap == cls_val] = remap_classes[cls_val]
        return remapped_labelmap

    def evaluate(self, pred_dir, gt_dir):
        img_cnt = 0
        for filename in os.listdir(pred_dir):
            print(filename)
            
            pred_path = os.path.join(pred_dir, filename)
            gt_path = os.path.join(gt_dir, filename)
            predmap = ImageHelper.img2np(ImageHelper.read_image(pred_path, tool='pil', mode='P'))
            gtmap = ImageHelper.img2np(ImageHelper.read_image(gt_path, tool='pil', mode='P'))

            # predmap = self.relabel(predmap)
            gtmap = self.relabel(gtmap)

            # fig, ax = plt.subplots(1, 2)
            # # ax[0].imshow(img, cmap='gray')
            # ax[0].imshow(predmap, cmap='seismic', alpha=0.65)
            # # ax[1].imshow(img, cmap='gray')
            # ax[1].imshow(gtmap, cmap='seismic', alpha=0.65)
            # plt.show()

            self.seg_running_score.update(predmap[np.newaxis, :, :], gtmap[np.newaxis, :, :])
            img_cnt += 1

        Log.info('Evaluate {} images'.format(img_cnt))
        Log.info('mIOU: {}'.format(self.seg_running_score.get_mean_iou()))
        Log.info('IOU-Variance: {}'.format(self.seg_running_score.get_std_iou()))
        Log.info('Pixel ACC: {}'.format(self.seg_running_score.get_pixel_acc()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', default=None, type=str,
                        dest='configs', help='The configs file of pose.')
    parser.add_argument('--gt_dir', default=None, type=str,
                        dest='gt_dir', help='The groundtruth annotations.')
    parser.add_argument('--pred_dir', default=None, type=str,
                        dest='pred_dir', help='The label dir of predict annotations.')
    args = parser.parse_args()

    pcontext_evaluator = ThermalFaceDBEvaluator(Configer(configs=args.configs))
    pcontext_evaluator.evaluate(args.pred_dir, args.gt_dir)
