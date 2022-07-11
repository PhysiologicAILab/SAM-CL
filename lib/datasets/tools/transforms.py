#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from PIL import Image


class Normalize(object):
    """Normalize a ``torch.tensor``

    Args:
        inputs (torch.tensor): tensor to be normalized.
        mean: (list): the mean of RGB
        std: (list): the std of RGB

    Returns:
        Tensor: Normalized tensor.
    """
    def __init__(self, div_value, mean, std):
        self.div_value = div_value
        self.mean = mean
        self.std =std

    def __call__(self, inputs):
        inputs = inputs.div(self.div_value)
        for t, m, s in zip(inputs, self.mean, self.std):
            t.sub_(m).div_(s)

        return inputs

class NormalizeThermal(object):
    """Normalize a ``torch.tensor``

    Args:
        inputs (torch.tensor): tensor to be normalized.
        norm_mode: (int): 1: normalize image with 0->1 range; 2: normalize image with -1->1 range 

    Returns:
        Tensor: Normalized tensor.
    """
    def __init__(self, norm_mode=1):
        self.norm_mode = norm_mode

    def __call__(self, inputs):

        min_T = inputs.min()
        max_T = inputs.max()

        if self.norm_mode == 1:
            if (max_T - min_T) != 0:
                inputs = (inputs - min_T) / (max_T - min_T)
            elif max_T != 0:
                inputs = inputs / max_T
            else:
                pass
        elif self.norm_mode == 2:
            if (max_T - min_T) != 0:
                inputs = 2*((inputs - min_T) / (max_T - min_T)) - 1
            elif max_T != 0:
                inputs = 2*(inputs / max_T) - 1
            else:
                pass
        else:
            print("Incorrect normalization mode found, images not normalized... Consider re-running with correct parameters...")

        return inputs


class DeNormalize(object):
    """DeNormalize a ``torch.tensor``

    Args:
        inputs (torch.tensor): tensor to be normalized.
        mean: (list): the mean of RGB
        std: (list): the std of RGB

    Returns:
        Tensor: Normalized tensor.
    """
    def __init__(self, div_value, mean, std):
        self.div_value = div_value
        self.mean = mean
        self.std =std

    def __call__(self, inputs):
        result = inputs.clone()
        for i in range(result.size(0)):
            result[i, :, :] = result[i, :, :] * self.std[i] + self.mean[i]

        return result.mul_(self.div_value)


class ToTensor(object):
    """Convert a ``numpy.ndarray or Image`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        inputs (numpy.ndarray or Image): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    def __call__(self, inputs):
        if isinstance(inputs, Image.Image):
            channels = len(inputs.mode)
            inputs = np.array(inputs)
            inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], channels)
            inputs = torch.from_numpy(inputs.transpose(2, 0, 1))
        else:
            if len(inputs.shape) == 2:
                inputs = np.expand_dims(inputs, axis=2)
            inputs = torch.from_numpy(inputs.transpose(2, 0, 1))

        return inputs.float()


class ToLabel(object):
    def __call__(self, inputs):
        return torch.from_numpy(np.array(inputs)).long()


class ReLabel(object):
    """
      255 indicate the background, relabel 255 to some value.
    """
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, inputs):
        assert isinstance(inputs, torch.LongTensor), 'tensor needs to be LongTensor'

        inputs[inputs == self.olabel] = self.nlabel
        return inputs


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inputs):
        for t in self.transforms:
            inputs = t(inputs)

        return inputs




