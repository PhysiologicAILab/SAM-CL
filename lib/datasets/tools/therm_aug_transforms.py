#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Jingyi Xie

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import random

import cv2
# from skimage.transform import resize
import numpy as np

from lib.utils.tools.logger import Logger as Log
from lib.datasets.tools.transforms import DeNormalize
from scipy.ndimage import rotate
from scipy.ndimage.filters import gaussian_filter
from .thermal_occlusions import ThermOcclusion

class _BaseTransform(object):
    DATA_ITEMS = (
        'labelmap', 'gcl_input'
    )

    def __call__(self, img, **kwargs):

        data_dict = collections.defaultdict(lambda: None)
        data_dict.update(kwargs)

        return img, data_dict

    def _process(self, img, data_dict, skip_condition, *args, **kwargs):
        assert isinstance(img, np.ndarray), \
            "img should be numpy array, got {}.".format(type(img))
        if not skip_condition:
            img = self._process_img(img, *args, **kwargs)

        ret_dict = collections.defaultdict(lambda: None)
        for name in self.DATA_ITEMS:
            func_name = '_process_' + name
            x = data_dict[name]

            assert isinstance(x, np.ndarray) or x is None, \
                "{} should be numpy array or None, got {}.".format(
                    name, type(x))

            if hasattr(self, func_name) and x is not None and not skip_condition:
                ret_dict[name] = getattr(self, func_name)(x, *args, **kwargs)
            else:
                ret_dict[name] = x

        return img, ret_dict


class Padding(_BaseTransform):
    """ Padding the Image to proper size.
            Args:
                stride: the stride of the network.
                pad_value: the value that pad to the image border.
                img: Image object as input.
            Returns::
                img: Image object.
    """

    def __init__(self, pad=None, ratio=0.5, mean=25.0, allow_outside_center=True):
        self.pad = pad
        self.ratio = ratio
        self.mean = mean
        self.allow_outside_center = allow_outside_center

    def _pad(self, x, pad_value, height, width, target_size, offset_left, offset_up):
        expand_x = np.zeros((max(height, target_size[1]) + abs(offset_up), max(width, target_size[0]) + abs(offset_left)), dtype=x.dtype)
        expand_x[:, :] = pad_value
        expand_x[abs(min(offset_up, 0)):abs(min(offset_up, 0)) + height, abs(min(offset_left, 0)):abs(min(offset_left, 0)) + width] = x
        x = expand_x[max(offset_up, 0):max(offset_up, 0) + target_size[1], max(offset_left, 0):max(offset_left, 0) + target_size[0]]
        return x

    def _process_img(self, img, *args):
        return self._pad(img, self.mean, *args)

    def _process_labelmap(self, x, *args):
        return self._pad(x, 0, *args)

    def _process_gcl_input(self, x, *args):
        return self._pad(x, 0, *args)

    def __call__(self, img, **kwargs):
        img, data_dict = super().__call__(img, **kwargs)

        height, width = img.shape
        left_pad, up_pad, right_pad, down_pad = self.pad
        target_size = [width + left_pad + right_pad, height + up_pad + down_pad]
        offset_left = -left_pad
        offset_up = -up_pad

        return self._process(img, data_dict, random.random() > self.ratio, height, width, target_size, offset_left, offset_up)
        

class RandomHFlip(_BaseTransform):
    def __init__(self, ratio):
        self.ratio = ratio

    def _process_img(self, x):
        return np.fliplr(x)

    def _process_gcl_input(self, x):
        return np.fliplr(x)

    def _process_labelmap(self, x):
        return np.fliplr(x)

    def __call__(self, img, **kwargs):
        img, data_dict = super().__call__(img, **kwargs)

        return self._process(img, data_dict, random.random() > self.ratio)

class RandomVFlip(_BaseTransform):
    def __init__(self, ratio):
        self.ratio = ratio

    def _process_img(self, x):
        return np.flipud(x)

    def _process_gcl_input(self, x):
        return np.flipud(x)

    def _process_labelmap(self, x):
        return np.flipud(x)

    def __call__(self, img, **kwargs):
        img, data_dict = super().__call__(img, **kwargs)

        return self._process(img, data_dict, random.random() > self.ratio)


class RandomRotate(_BaseTransform):
    """Rotate the input numpy.ndarray and points to the given degree.

    Args:
        limits of angle (number): limits for degree of rotation.
    """

    def __init__(self, ratio, low_limit_angle=5, high_limit_angle=355):

        self.ratio = ratio
        self.low_limit_angle = low_limit_angle
        self.high_limit_angle = high_limit_angle

    def _rotate(self, x, rotation_angle):
        return rotate(x, rotation_angle, reshape=False, mode="nearest", order=3)

    def _rotate_map(self, x, rotation_angle):
        rotated_x = np.zeros(x.shape)
        x = x.astype(np.float)
        num_classes = int(np.max(x)) + 1
        for i in range(1, num_classes):
            cls_mask = np.zeros(x.shape)
            cls_mask[x == i] = 1
            cls_mask = rotate(cls_mask, rotation_angle, reshape=False, mode="constant", cval=np.min(cls_mask), order=3)
            rotated_x[cls_mask >= 0.33] = i
        rotated_x = rotated_x.astype(np.uint8)
        return rotated_x

    def _process_img(self, x):
        return self._rotate(x, self.rotation_angle)

    def _process_gcl_input(self, x):
        return self._rotate(x, self.rotation_angle)

    def _process_labelmap(self, x):
        return self._rotate_map(x, self.rotation_angle)

    def __call__(self, img, **kwargs):
        """rotation_angle
        Args:
            img    (Image):     Image to be rotated.
            maskmap   (Image):     Mask to be rotated.
            kpt    (list):      Keypoints to be rotated.
            center (list):      Center points to be rotated.

        Returns:
            Image:     Rotated image.
            list:      Rotated key points.
        """
        img, data_dict = super().__call__(img, **kwargs)

        self.rotation_angle = np.random.randint(self.low_limit_angle, self.high_limit_angle)

        return self._process(img, data_dict, random.random() > self.ratio)


class GaussianBlur(_BaseTransform):

    def __init__(self, ratio, blur_sigma_max=1.2):
        self.ratio = ratio
        self.blur_sigma_max = blur_sigma_max

    def _gaussian_blur(self, x, blur_sigma):
        return gaussian_filter(x, sigma=blur_sigma)

    def _process_img(self, x):
        return self._gaussian_blur(x, self.blur_sigma)

    def __call__(self, img, **kwargs):
        """blur_sigma
        Args:
            img    (Image):     Image to be blurred.
        Returns:
            Image:     Blurred image.
        """
        img, data_dict = super().__call__(img, **kwargs)
        self.blur_sigma = np.random.uniform(1, self.blur_sigma_max)
        return self._process(img, data_dict, random.random() > self.ratio)


class ThermalNoise(_BaseTransform):

    def __init__(self, ratio, max_noise_equivalent_differential_temperature=0.1):
        self.ratio = ratio
        self.max_nedt = max_noise_equivalent_differential_temperature

    def _add_noise(self, x, nedt):
        return x + nedt * (np.random.random(x.shape) - 0.5)

    def _process_img(self, x):
        return self._add_noise(x, self.nedt)

    def __call__(self, img, **kwargs):
        """blur_sigma
        Args:
            img    (Image):     Image to be blurred.
        Returns:
            Image:     Blurred image.
        """
        img, data_dict = super().__call__(img, **kwargs)
        self.nedt = np.random.uniform(0, self.max_nedt)
        return self._process(img, data_dict, random.random() > self.ratio)


class RandomResize(_BaseTransform):
    """Resize the given numpy.ndarray to random size and aspect ratio.

    Args:
        scale_min: the min scale to resize.
        scale_max: the max scale to resize.
    """

    def __init__(self, scale_range=(0.25, 2.0), aspect_range=(0.9, 1.1), target_size=None,
                 resize_bound=None, method='random', max_side_bound=None, scale_list=None, ratio=0.5):
        self.scale_range = scale_range
        self.aspect_range = aspect_range
        self.resize_bound = resize_bound
        self.max_side_bound = max_side_bound
        self.scale_list = scale_list
        self.method = method
        self.ratio = ratio

        if target_size is not None:
            if isinstance(target_size, int):
                self.input_size = (target_size, target_size)
            elif isinstance(target_size, (list, tuple)) and len(target_size) == 2:
                self.input_size = target_size
            else:
                raise TypeError(
                    'Got inappropriate size arg: {}'.format(target_size))
        else:
            self.input_size = None

    def get_scale(self, img_size):
        if self.method == 'random':
            scale_ratio = random.uniform(self.scale_range[0], self.scale_range[1])
            return scale_ratio

        elif self.method == 'bound':
            scale1 = self.resize_bound[0] / min(img_size)
            scale2 = self.resize_bound[1] / max(img_size)
            scale = min(scale1, scale2)
            return scale

        else:
            Log.error('Resize method {} is invalid.'.format(self.method))
            exit(1)

    def _process_img(self, x, converted_size, *args):
        # return resize(x, converted_size)
        return cv2.resize(x, converted_size, interpolation=cv2.INTER_CUBIC)

    def _process_gcl_input(self, x, converted_size, *args):
        # return resize(x, converted_size)
        return cv2.resize(x, converted_size, interpolation=cv2.INTER_CUBIC)

    def _process_labelmap(self, x, converted_size, *args):

        # n_classes = int(x.max() + 1)
        # mask_canvas = np.zeros(shape=converted_size)

        # # To avoid misclassification at the boundaries
        # for i in range(n_classes):
        #     cls_mask = np.zeros(shape=x.shape)
        #     cls_mask[(x == i)] = i
        #     cls_mask = resize(cls_mask, converted_size)
        #     mask_canvas[cls_mask > 0] = i

        # return mask_canvas
        return cv2.resize(x, converted_size, interpolation=cv2.INTER_NEAREST)

    def __call__(self, img, **kwargs):
        """
        Args:
            img     (Image):   Image to be resized.
            maskmap    (Image):   Mask to be resized.
            kpt     (list):    keypoints to be resized.
            center: (list):    center points to be resized.

        Returns:
            Image:  Randomly resize image.
            Image:  Randomly resize maskmap.
            list:   Randomly resize keypoints.
            list:   Randomly resize center points.
        """
        img, data_dict = super().__call__(img, **kwargs)

        height, width = img.shape
        if self.scale_list is None:
            scale_ratio = self.get_scale([width, height])
        else:
            scale_ratio = self.scale_list[random.randint(
                0, len(self.scale_list) - 1)]

        aspect_ratio = random.uniform(*self.aspect_range)
        w_scale_ratio = math.sqrt(aspect_ratio) * scale_ratio
        h_scale_ratio = math.sqrt(1.0 / aspect_ratio) * scale_ratio
        if self.max_side_bound is not None and max(height * h_scale_ratio, width * w_scale_ratio) > self.max_side_bound:
            d_ratio = self.max_side_bound / max(height * h_scale_ratio, width * w_scale_ratio)
            w_scale_ratio *= d_ratio
            h_scale_ratio *= d_ratio

        converted_size = (int(width * w_scale_ratio), int(height * h_scale_ratio))
        return self._process(
            img, data_dict,
            random.random() > self.ratio,
            converted_size, h_scale_ratio, w_scale_ratio
        )


class RandomCrop(_BaseTransform):
    """Crop the given numpy.ndarray and  at a random location.

    Args:
        size (int or tuple): Desired output size of the crop.(w, h)
    """

    def __init__(self, crop_size, ratio=0.5, method='random', grid=None, allow_outside_center=True):
        self.ratio = ratio
        self.method = method
        self.grid = grid
        self.allow_outside_center = allow_outside_center

        if isinstance(crop_size, float):
            self.size = (crop_size, crop_size)
        elif isinstance(crop_size, collections.abc.Iterable) and len(crop_size) == 2:
            self.size = crop_size
        else:
            raise TypeError('Got inappropriate size arg: {}'.format(crop_size))

    def get_lefttop(self, crop_size, img_size):
        if self.method == 'center':
            return [(img_size[0] - crop_size[0]) // 2, (img_size[1] - crop_size[1]) // 2]

        elif self.method == 'random':
            x = random.randint(0, img_size[0] - crop_size[0])
            y = random.randint(0, img_size[1] - crop_size[1])
            return [x, y]

        elif self.method == 'grid':
            grid_x = random.randint(0, self.grid[0] - 1)
            grid_y = random.randint(0, self.grid[1] - 1)
            x = grid_x * ((img_size[0] - crop_size[0]) // (self.grid[0] - 1))
            y = grid_y * ((img_size[1] - crop_size[1]) // (self.grid[1] - 1))
            return [x, y]

        else:
            Log.error('Crop method {} is invalid.'.format(self.method))
            exit(1)

    def _crop(self, x, offset_up, offset_left, target_size):
        return x[offset_up:offset_up + target_size[1], offset_left:offset_left + target_size[0]]

    def _process_img(self, x, *args):
        return self._crop(x, *args)

    def _process_gcl_input(self, x, *args):
        return self._crop(x, *args)

    def _process_labelmap(self, x, *args):
        return self._crop(x, *args)


    def __call__(self, img, **kwargs):
        """
        Args:
            img (Image):   Image to be cropped.
            maskmap (Image):  Mask to be cropped.

        Returns:
            Image:  Cropped image.
            Image:  Cropped maskmap.
            list:   Cropped keypoints.
            list:   Cropped center points.
        """
        img, data_dict = super().__call__(img, **kwargs)

        height, width = img.shape
        target_size = [min(self.size[0], width), min(self.size[1], height)]

        offset_left, offset_up = self.get_lefttop(target_size, [width, height])
        return self._process(
            img, data_dict,
            random.random() > self.ratio,
            offset_up, offset_left, target_size
        )


class RandomThermalOcclusion(_BaseTransform):
    def __init__(self, ratio, max_noise_equivalent_differential_temperature=0.1):
        self.thermOccObj = ThermOcclusion()
        self.max_nedt = max_noise_equivalent_differential_temperature
        self.ratio = ratio

    def _process_img(self, x):
        return self.thermOccObj.gen_occluded_image(x, self.low_temp, self.high_temp, self.nedt_1, self.nedt_2)

    def _process_labelmap(self, x):
        return self.thermOccObj.gen_occluded_label(x)

    def __call__(self, img, **kwargs):
        img, data_dict = super().__call__(img, **kwargs)
        
        min_temp = np.min(img)
        max_temp = np.max(img)
        diff_temp = np.abs(max_temp - min_temp)
        self.low_temp = min_temp - diff_temp
        self.high_temp = max_temp + diff_temp
        
        self.nedt_1 = np.random.uniform(0, self.max_nedt)
        self.nedt_2 = np.random.uniform(0, self.max_nedt)
        
        return self._process(img, data_dict, random.random() > self.ratio)



class ThermAugCompose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> ThermAugCompose([
        >>>     RandomCrop(),
        >>> ])
    """

    def __init__(self, configer, split='train'):
        self.configer = configer
        self.split = split

        if self.split == 'train':
            shuffle_train_trans = []
            if self.configer.exists('train_trans', 'shuffle_trans_seq'):
                if isinstance(self.configer.get('train_trans', 'shuffle_trans_seq')[0], list):
                    train_trans_seq_list = self.configer.get(
                        'train_trans', 'shuffle_trans_seq')
                    for train_trans_seq in train_trans_seq_list:
                        shuffle_train_trans += train_trans_seq

                else:
                    shuffle_train_trans = self.configer.get(
                        'train_trans', 'shuffle_trans_seq')
            trans_seq = self.configer.get(
                'train_trans', 'trans_seq') + shuffle_train_trans
            trans_key = 'train_trans'
        else:
            trans_seq = self.configer.get('val_trans', 'trans_seq')
            trans_key = 'val_trans'

        self.transforms = dict()
        self.trans_config = self.configer.get(trans_key)
        for trans_name in trans_seq:
            specs = TRANSFORM_SPEC[trans_name]
            config = self.configer.get(trans_key, trans_name)
            for spec in specs:
                if 'when' not in spec:
                    break
                choose_this = True
                for cond_key, cond_value in spec['when'].items():
                    choose_this = choose_this and (
                            config[cond_key] == cond_value)
                if choose_this:
                    break
            else:
                raise RuntimeError("Not support!")

            kwargs = {}
            for arg_name, arg_path in spec["args"].items():
                if isinstance(arg_path, str):
                    arg_value = config.get(arg_path, None)
                elif isinstance(arg_path, list):
                    arg_value = self.configer.get(*arg_path)
                kwargs[arg_name] = arg_value

            klass = TRANSFORM_MAPPING[trans_name]
            self.transforms[trans_name] = klass(**kwargs)

    def __call__(self, img, **data_dict):

        orig_key_list = list(data_dict)

        if self.configer.get('data', 'input_mode') == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if self.split == 'train':
            shuffle_trans_seq = []
            if self.configer.exists('train_trans', 'shuffle_trans_seq'):
                if isinstance(self.configer.get('train_trans', 'shuffle_trans_seq')[0], list):
                    shuffle_trans_seq_list = self.configer.get('train_trans', 'shuffle_trans_seq')
                    shuffle_trans_seq = shuffle_trans_seq_list[random.randint(0, len(shuffle_trans_seq_list))]
                else:
                    shuffle_trans_seq = self.configer.get('train_trans', 'shuffle_trans_seq')
                    random.shuffle(shuffle_trans_seq)
            trans_seq = shuffle_trans_seq + self.configer.get('train_trans', 'trans_seq')
        else:
            trans_seq = self.configer.get('val_trans', 'trans_seq')

        for trans_key in trans_seq:
            img, data_dict = self.transforms[trans_key](img, **data_dict)

        if self.configer.get('data', 'input_mode') == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return (img, *[data_dict[key] for key in orig_key_list])

    def __repr__(self):
        import pprint
        return 'ThermAugCompose({})'.format(pprint.pformat(self.trans_config))


TRANSFORM_MAPPING = {
    "padding": Padding,
    "random_hflip": RandomHFlip,
    "random_vflip": RandomVFlip,
    "gaussian_blur": GaussianBlur,
    "random_resize": RandomResize,
    "random_crop": RandomCrop,
    "random_rotate": RandomRotate,
    "thermal_noise": ThermalNoise,
    "random_thermal_occlusion": RandomThermalOcclusion,
}

TRANSFORM_SPEC = {
    "random_thermal_contrast": [{
        "args": {
            "lower_temperature_limit": "lower_temperature_limit",
            "higher_temperature_limit": "higher_temperature_limit"
        }
    }],
    "padding": [{
        "args": {
            "pad": "pad",
            "ratio": "ratio",
            "mean": ["normalize", "mean_value"],
            "allow_outside_center": "allow_outside_center"
        }
    }],
    "random_hflip": [{
        "args": {
            "ratio": "ratio"
        }
    }],
    "random_vflip": [{
        "args": {
            "ratio": "ratio"
        }
    }],
    "random_resize": [
        {
            "args": {
                "method": "method",
                "scale_range": "scale_range",
                "aspect_range": "aspect_range",
                "max_side_bound": "max_side_bound",
                "ratio": "ratio"
            },
            "when": {
                "method": "random"
            }
        },
        {
            "args": {
                "method": "method",
                "scale_range": "scale_range",
                "aspect_range": "aspect_range",
                "target_size": "target_size",
                "ratio": "ratio"
            },
            "when": {
                "method": "focus"
            }
        },
        {
            "args": {
                "method": "method",
                "aspect_range": "aspect_range",
                "resize_bound": "resize_bound",
                "ratio": "ratio"
            },
            "when": {
                "method": "bound"
            }
        },
    ],
    "random_crop": [
        {
            "args": {
                "crop_size": "crop_size",
                "method": "method",
                "ratio": "ratio",
                "allow_outside_center": "allow_outside_center"
            },
            "when": {
                "method": "random"
            }
        },
        {
            "args": {
                "crop_size": "crop_size",
                "method": "method",
                "ratio": "ratio",
                "allow_outside_center": "allow_outside_center"
            },
            "when": {
                "method": "center"
            }
        },
        {
            "args": {
                "crop_size": "crop_size",
                "method": "method",
                "ratio": "ratio",
                "grid": "grid",
                "allow_outside_center": "allow_outside_center"
            },
            "when": {
                "method": "grid"
            }
        },
    ],
    "random_rotate": [{
        "args": {
            "ratio": "ratio",
            "low_limit_angle": "low_limit_angle",
            "high_limit_angle": "high_limit_angle"
        }
    }],
    "gaussian_blur": [{
        "args": {
            "ratio": "ratio",
            "blur_sigma_max": "blur_sigma_max",
        }
    }],
    "thermal_noise": [{
        "args": {
            "ratio": "ratio",
            "max_noise_equivalent_differential_temperature": "max_noise_equivalent_differential_temperature",
        }
    }],
    "random_thermal_occlusion": [{
        "args": {
            "ratio": "ratio",
            "max_noise_equivalent_differential_temperature": "max_noise_equivalent_differential_temperature",
        }
    }],
    "resize": [{
        "args": {
            "target_size": "target_size",
            "min_side_length": "min_side_length",
            "max_side_bound": "max_side_bound",
            "max_side_length": "max_side_length"
        }
    }],
}
