##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: JingyiXie, LangHuang, DonnyYou, RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from copy import deepcopy

import os
import pdb

import numpy as np
from torch.utils import data

from lib.utils.helpers.image_helper import ImageHelper
from lib.extensions.parallel.data_container import DataContainer
from lib.utils.tools.logger import Logger as Log


class ThermalFaceDBLoader(data.Dataset):
    def __init__(self, root_dir, aug_transform=None, dataset=None,
                 img_transform=None, label_transform=None, configer=None):
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.img_list, self.label_list, self.name_list = self.__list_dirs(root_dir, dataset)
        size_mode = self.configer.get(dataset, 'data_transformer')['size_mode']
        self.is_stack = size_mode != 'diverse_size'

        Log.info('{} {}'.format(dataset, len(self.img_list)))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = ImageHelper.read_image(self.img_list[index],
                                     tool=self.configer.get('data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))
        # Log.info('{}'.format(self.img_list[index]))
        img_size = ImageHelper.get_size(img)
        labelmap = ImageHelper.read_image(self.label_list[index],
                                          tool=self.configer.get('data', 'image_tool'), mode='P')

        if self.configer.exists('data', 'remap_classes'):
            labelmap = self._remap_classes(labelmap, self.configer.get('data', 'remap_classes'))

        ori_target = ImageHelper.tonp(labelmap)

        if self.aug_transform is not None:
            img, labelmap = self.aug_transform(img, labelmap=labelmap)

        border_size = ImageHelper.get_size(img)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            labelmap = self.label_transform(labelmap)

        meta = dict(
            ori_img_size=img_size,
            border_size=border_size,
            ori_target=ori_target
        )
        Log.info('Labelmap Min Max: {} {}'.format(labelmap.min(), labelmap.max()))
        
        return_dict = dict(
            img=DataContainer(img, stack=self.is_stack),
            labelmap=DataContainer(labelmap, stack=self.is_stack),
            meta=DataContainer(meta, stack=False, cpu_only=True),
            name=DataContainer(self.name_list[index], stack=False, cpu_only=True),
        )

        # Log.info('return_dict: Labelmap Min Max: {} {}'.format(
        #     return_dict['labelmap'].min(), return_dict['labelmap'].max()))

        return return_dict

    def _remap_classes(self, labelmap, remap_classes):
        max_cls_val = np.max(labelmap)
        remapped_labelmap = deepcopy(labelmap)
        for cls_val in range(max_cls_val+1):
            remapped_labelmap[labelmap==cls_val] = remap_classes[cls_val]
        return remapped_labelmap

    def __list_dirs(self, root_dir, dataset):
        img_list = list()
        label_list = list()
        name_list = list()
        image_dir = os.path.join(root_dir, dataset, 'image')
        label_dir = os.path.join(root_dir, dataset, 'label')

        img_extension = os.listdir(image_dir)[0].split('.')[-1]

        # support the argument to pass the file list used for training/testing
        file_list_txt = os.environ.get('use_file_list')
        if file_list_txt is None:
            files = sorted(os.listdir(image_dir))
        else:
            Log.info("Using file list {} for training".format(file_list_txt))
            with open(os.path.join(root_dir, dataset, 'file_list', file_list_txt)) as f:
                files = [x.strip() for x in f]

        for file_name in files:
            image_name = '.'.join(file_name.split('.')[:-1])
            img_path = os.path.join(image_dir, '{}'.format(file_name))
            label_path = os.path.join(label_dir, image_name + '.png')
            # Log.info('{} {} {}'.format(image_name, img_path, label_path))
            if not os.path.exists(label_path) or not os.path.exists(img_path):
                Log.error('Label Path: {} {} not exists.'.format(label_path, img_path))
                continue

            img_list.append(img_path)
            label_list.append(label_path)
            name_list.append(image_name)

        return img_list, label_list, name_list



if __name__ == "__main__":
    # Test ThermalFaceDB loader.
    pass
