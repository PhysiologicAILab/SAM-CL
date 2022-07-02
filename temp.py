import numpy as np
import os
import cv2
import copy

'''
# pth = "/home/uclic/dev/data/thermal/Thermal_FaceDB/Processed/val/label/"
pth = "/home/uclic/dev/data/thermal/Thermal_FaceDB/Processed/train/label/"

flist = os.listdir(pth)

src_extension = '.npy'
target_extension = '.png'

for fn in flist:
    img_extension = fn.split('.')[-1]
    if img_extension in src_extension:
        src_path = os.path.join(pth, fn)
        img = np.load(src_path).astype(np.uint8)
        target_fn = fn.replace(src_extension, target_extension)
        dst_path = os.path.join(pth, target_fn)
        print("Saving...", target_fn)
        cv2.imwrite(dst_path, img)

'''
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

pth = "/home/uclic/dev/data/thermal/Thermal_FaceDB/Processed/train/label/"
flist = os.listdir(pth)
remap_classes = [0, 1, 2, 3, 3, 4, 4, 5]

for fn in flist:
    labelmap = cv2.imread(os.path.join(pth, fn), 0)

    max_cls_val = np.max(labelmap)
    remapped_labelmap = copy.deepcopy(labelmap)
    for cls_val in range(max_cls_val+1):
        remapped_labelmap[labelmap == cls_val] = remap_classes[cls_val]

    plt.imshow(remapped_labelmap)
    plt.show()
    break