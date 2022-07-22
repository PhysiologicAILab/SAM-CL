import argparse
import os
import sys
import shutil
import glob
import numpy as np
from PIL import Image

from val_ids import validation_ids
from generate_masks_from_landmarks import GenerateMask


def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


def main(args):

	ori_root_dir = args.ori_root_dir
	if not os.path.exists(ori_root_dir):
		print("Data path not found:", ori_root_dir)
		sys.exit(0)

	save_dir = args.save_dir
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
		print("Creating directories at:", save_dir)

	crop_res = tuple(args.crop_res)
	if crop_res[0] <= 0:
		crop_res = (768, 960)
		print('invalid crop resolution specified, cropping to default', crop_res)

	target_res = tuple(args.target_res)
	if target_res[0] <= 0:
		target_res = (256, 320)
		print('invalid target resolution specified, resizing to default', target_res)

	genMaskObj = GenerateMask(crop_res=crop_res, target_res=target_res)

	save_dir_train_image = os.path.join(save_dir, "train", "image")
	if not os.path.exists(save_dir_train_image):
		os.makedirs(save_dir_train_image)
	save_dir_train_label = os.path.join(save_dir, "train", "label")
	if not os.path.exists(save_dir_train_label):
		os.makedirs(save_dir_train_label)
	save_dir_train_ann = os.path.join(save_dir, "train", "ann")
	if not os.path.exists(save_dir_train_ann):
		os.makedirs(save_dir_train_ann)

	save_dir_valid_image = os.path.join(save_dir, "val", "image")
	if not os.path.exists(save_dir_valid_image):
		os.makedirs(save_dir_valid_image)
	save_dir_valid_label = os.path.join(save_dir, "val", "label")
	if not os.path.exists(save_dir_valid_label):
		os.makedirs(save_dir_valid_label)
	save_dir_valid_ann = os.path.join(save_dir, "val", "ann")
	if not os.path.exists(save_dir_valid_ann):
		os.makedirs(save_dir_valid_ann)

	ljson_FilenamesList = glob.glob(os.path.join(ori_root_dir, '*.ljson'))
	total_files = len(ljson_FilenamesList)
	print("Total Files Found:", total_files)
	print("")
	# print(ljson_FilenamesList[0])

	img_ext = '.npy'
	ann_ext = '.ljson'
	label_ext = '.png'

	for i in range(total_files):
		fn = ljson_FilenamesList[i]
		sys.stdout.write("Processing: " + str(i) + " of " + str(total_files) + ": " + fn + "\r")
		sys.stdout.flush()

		ext = os.path.splitext(fn)[-1]
		base_name_generic = os.path.basename(fn).replace(ext, '')
		target_base_name_generic = base_name_generic.replace('.jpg_lfb', '').replace('.png_lfb', '')

		sub_id = base_name_generic.split("_")[1]
		if sub_id in validation_ids:
			img_dir = save_dir_valid_image
			label_dir = save_dir_valid_label
			ann_dir = save_dir_valid_ann
		else:
			img_dir = save_dir_train_image
			label_dir = save_dir_train_label
			ann_dir = save_dir_train_ann

		src_img_fname = os.path.join(ori_root_dir, base_name_generic + img_ext)
		dst_img_fname = os.path.join(save_dir, img_dir, target_base_name_generic + img_ext)

		src_ann_fname = os.path.join(ori_root_dir, base_name_generic + ann_ext)
		dst_ann_fname = os.path.join(save_dir, ann_dir, target_base_name_generic + ann_ext)
		shutil.copyfile(src_ann_fname, dst_ann_fname)

		dst_label_fname = os.path.join(save_dir, label_dir, target_base_name_generic + label_ext)
		
		if (crop_res != None) or (target_res != None):
			label, img = genMaskObj.generate_roi_mask(src_img_fname, src_ann_fname)
			np.save(dst_img_fname, img)
		else:
			label = genMaskObj.generate_roi_mask(src_img_fname, src_ann_fname)
			shutil.copyfile(src_img_fname, dst_img_fname)

		label = Image.fromarray(label)
		label.save(dst_label_fname)

	print("\nPre-processing completed successfully...\n")


def parse_args():
	parser = argparse.ArgumentParser(description='Preprocess Thermal Face Database')
	parser.add_argument('--ori_root_dir', type=str)
	parser.add_argument('--save_dir', type=str)
	parser.add_argument('--crop_res', type=tuple_type, default=(768, 960))
	parser.add_argument('--target_res', type=tuple_type, default=(256, 320))

	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = parse_args()
	main(args)
