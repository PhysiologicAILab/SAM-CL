import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

def main(args):
   
    test_data_dir = args.img_dir
    in_files = []
    in_ext = ['.npy']

    test_data_dir_list = os.listdir(test_data_dir)
    # random.shuffle(test_data_dir_list)
    print("test_data_dir_list[0]", test_data_dir_list[0])
    print("in_ext[[d in test_data_dir_list[0] for d in in_ext]", [d in test_data_dir_list[0] for d in in_ext])

    in_ext = in_ext[[d in test_data_dir_list[0] for d in in_ext].index(True)]

    for ff in range(len(test_data_dir_list)):
        # fname = "frm_" + str(ff) + in_ext
        fname = test_data_dir_list[ff]
        in_files.append(os.path.join(test_data_dir, fname))

    for i, fn in enumerate(in_files):
        try:
            if in_ext in in_files[i]: # and i < 50:                

                input_img = np.load(in_files[i])
                # input_img[input_img > 100] = 50
                # x0, y0, x1, y1 = 32, 0, input_img.shape[0]-32, input_img.shape[1]                
                # input_img = input_img[x0:x1, y0:y1]

                col = (np.random.random(), np.random.random(), np.random.random())
                # col = (0.5, 0.5, 0.5)
                hist_im, bin_edges = np.histogram(input_img, bins=1024, range=(0, 40))
                plt.plot(bin_edges[0:-1], hist_im, color=col)

        except KeyboardInterrupt:
            return

        # break

    plt.show()


def get_args():
    parser = argparse.ArgumentParser(description='Generate the histogram plot of thermal matrix',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--img_dir', type=str,
                        help='Image Directory', dest='img_dir')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)
