import numpy as np
import json
import cv2
from skimage.transform import resize

class GenerateMask():
	def __init__(self, image_height=None, target_res=None) -> None:
		# self.labels = ['nose', 'mouth', 'eye', 'eyebrow']
		# self.class_val = [1, 2, 3, 3, 4, 4, 5]
		self.labels = ['chin', 'mouth', 'leye', 'reye', 'leyebrow', 'reyebrow', 'nose']
		self.class_val = [1, 2, 3, 4, 5, 6, 7]
		if image_height != None:
			self.crop_res = int(float(image_height)/2.0)
		else:
			self.crop_res = None
		self.target_res = target_res

	def crop_resize_mask(self, img, mask, landmarks_dict):
		n_classes = int(mask.max() + 1)
		height, width = img.shape
		
		# print(len(landmarks_dict['landmarks']['points']))
		pt_arr = np.array(landmarks_dict['landmarks']['points'])
		# print(pt_arr.shape)
		ymin = int(np.min(pt_arr, 0)[0])
		xmin = int(np.min(pt_arr, 0)[1])
		ymax = int(np.max(pt_arr, 0)[0])
		xmax = int(np.max(pt_arr, 0)[1])
		mid_x = int((xmax + xmin)/2.0)
		# mid_y = int((ymax + ymin)/2.0)

		if self.crop_res != None:
			if mid_x - self.crop_res >= 0:
				if mid_x + self.crop_res < width:
					crop_xmin = mid_x - self.crop_res
					crop_xmax = mid_x + self.crop_res
				else:
					crop_xmin = mid_x - self.crop_res - ((mid_x + self.crop_res) - width)
					crop_xmax = width
			else:
				crop_xmin = 0
				crop_xmax = mid_x + self.crop_res + (self.crop_res - mid_x)

			img = img[:, crop_xmin:crop_xmax]
			mask = mask[:, crop_xmin:crop_xmax]

		mask_canvas = np.zeros(shape=self.target_res)

		# To avoid misclassification at the boundaries
		for i in range(n_classes):
			cls_mask = np.zeros(shape=mask.shape)
			cls_mask[(mask == i)] = i
			cls_mask = resize(cls_mask, self.target_res)
			mask_canvas[cls_mask > 0] = i

		img = resize(img, self.target_res)
		return mask_canvas, img


	def generate_roi_mask(self, img_path, ann_path):

		img = np.load(img_path)
		with open(ann_path, 'r') as f:
			data = json.loads(f.read())
			f.close()

		landmarks_dict = dict(data)
		mask = np.zeros_like(img)

		points_dict = {}
		for item in landmarks_dict['labels']:
			if item['label'] in ['centernose', 'bottomnose', 'leyebrow', 'reyebrow', 'leye', 'reye', 'outermouth', 'innermouth', 'chin']:
				lbl = item['label']
				if lbl in ['centernose', 'bottomnose']:
					common_lbl_name = 'nose'
				elif lbl in ['outermouth', 'innermouth']:
					common_lbl_name = 'mouth'
				elif lbl in ['leyebrow']:
					common_lbl_name = 'leyebrow'
					# common_lbl_name = 'leye'
				elif lbl in ['leye']:
					common_lbl_name = 'leye'
				elif lbl in ['reye']:
					common_lbl_name = 'reye'
				elif lbl in ['reyebrow']:
					common_lbl_name = 'reyebrow'
					# common_lbl_name = 'reye'
				elif lbl in ['chin']:
					common_lbl_name = 'chin'
				else:
					# common_lbl_name = lbl
					pass
				if common_lbl_name not in points_dict:
					points_dict[common_lbl_name] = []
				msk = item['mask']
				for val in msk:
					y = int(landmarks_dict['landmarks']['points'][val][0])
					x = int(landmarks_dict['landmarks']['points'][val][1])
					points_dict[common_lbl_name].append([x, y])

		for lbl in self.labels:
			pts = points_dict[lbl]
			cls_val = 255
			pts_arr = np.array(pts)
			hull = cv2.convexHull(pts_arr)
			mask = cv2.fillConvexPoly(mask, hull, cls_val)
			cls_val = self.class_val[self.labels.index(lbl)]
			mask[mask >= 100] = cls_val

		if (self.crop_res != None) or (self.target_res != None):
			mask, img = self.crop_resize_mask(img, mask, landmarks_dict)
			mask = mask.astype(np.uint8)
			return mask, img
		else:
			mask = mask.astype(np.uint8)
			return mask