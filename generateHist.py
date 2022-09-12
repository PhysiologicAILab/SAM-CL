import numpy as np
import cv2
import argparse
import os
import matplotlib.pyplot as plt

def main(args):
   
    data_dir = args.img_dir
    label_dir = args.label_dir
    save_dir = args.save_dir
    mode =int(args.mode)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    img_files = []
    label_files = []
    img_ext = ['.npy']
    label_ext = ['.png']

    data_dir_list = os.listdir(data_dir)
    # random.shuffle(data_dir_list)
    print("data_dir_list[0]", data_dir_list[0])
    print("img_ext[[d in data_dir_list[0] for d in img_ext]", [d in data_dir_list[0] for d in img_ext])

    img_ext = img_ext[[d in data_dir_list[0] for d in img_ext].index(True)]

    for ff in range(len(data_dir_list)):
        # fname = "frm_" + str(ff) + img_ext
        fname = data_dir_list[ff]
        img_files.append(os.path.join(data_dir, fname))
        label_files.append(os.path.join(label_dir, fname.replace(img_ext, label_ext)))

    min_list = []
    avg_list = []
    max_list = []
    std_list = []
    fg_avg_list = []
    bg_avg_list = []

    for i, fn in enumerate(img_files):
        try:
            if img_ext in img_files[i]: # and i < 50:                

                input_img = np.load(img_files[i])
                label_img = cv2.imread(label_files[i], 0)
                if mode == 1:
                    input_img = (input_img + 1) * 20
                elif mode == 2:
                    input_img = input_img/ 1000.0
                
                # input_img[input_img > 50] = 50
                # x0, y0, x1, y1 = 32, 0, input_img.shape[0]-32, input_img.shape[1]                
                # input_img = input_img[x0:x1, y0:y1]

                min_list.append(np.min(input_img))
                avg_list.append(np.mean(input_img))
                max_list.append(np.max(input_img))
                std_list.append(np.std(input_img))
                
                # print('type of input_img', type(input_img), type(input_img[0, 0]))
                # print('type of label_img', type(label_img), type(label_img[0, 0]))
                fg_avg_temp = np.mean(input_img[label_img > 0])
                bg_avg_temp = np.mean(input_img[label_img == 0])
                fg_avg_list.append(fg_avg_temp)
                bg_avg_list.append(bg_avg_temp)
                
                col = (np.random.random(), np.random.random(), np.random.random())
                # col = (0.5, 0.5, 0.5)
                hist_im, bin_edges = np.histogram(input_img, bins=1024, range=(0, 40))
                plt.plot(bin_edges[0:-1], hist_im, color=col)

        except KeyboardInterrupt:
            return

        # break

    # plt.show()
    plt.savefig(os.path.join(save_dir, 'histogram.jpg'), bbox_inches=0)

    min_list = np.array(min_list)
    avg_list = np.array(avg_list)
    max_list = np.array(max_list)
    std_list = np.array(std_list)
    fg_avg_list = np.array(fg_avg_list)
    bg_avg_list = np.array(bg_avg_list)

    np.save(os.path.join(save_dir, 'min_array.npy'), min_list)
    np.save(os.path.join(save_dir, 'avg_array.npy'), avg_list)
    np.save(os.path.join(save_dir, 'max_array.npy'), max_list)
    np.save(os.path.join(save_dir, 'std_array.npy'), std_list)

    np.save(os.path.join(save_dir, 'fg_avg_array.npy'), fg_avg_list)
    np.save(os.path.join(save_dir, 'bg_avg_array.npy'), bg_avg_list)


def get_args():
    parser = argparse.ArgumentParser(description='Generate the histogram plot of thermal matrix',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--img_dir', type=str,
                        help='Image Directory', dest='img_dir')
    parser.add_argument('-l', '--label_dir', type=str,
                        help='Label Directory', dest='label_dir')
    parser.add_argument('-m', '--mode', type=int, default=0,
                        help='Normalization Mode', dest='mode')
    parser.add_argument('-o', '--out_dir', type=str,
                        help='Output Directory', dest='save_dir')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)
