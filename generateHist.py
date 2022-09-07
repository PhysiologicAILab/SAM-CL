import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

def main(args):
   
    test_data_dir = args.img_dir
    save_dir = args.save_dir
    mode =int(args.mode)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
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

    min_list = []
    avg_list = []
    max_list = []
    std_list = []

    for i, fn in enumerate(in_files):
        try:
            if in_ext in in_files[i]: # and i < 50:                

                input_img = np.load(in_files[i])
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

    np.save(os.path.join(save_dir, 'min_array.npy'), min_list)
    np.save(os.path.join(save_dir, 'avg_array.npy'), avg_list)
    np.save(os.path.join(save_dir, 'max_array.npy'), max_list)
    np.save(os.path.join(save_dir, 'std_array.npy'), std_list)


def get_args():
    parser = argparse.ArgumentParser(description='Generate the histogram plot of thermal matrix',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--img_dir', type=str,
                        help='Image Directory', dest='img_dir')
    parser.add_argument('-m', '--mode', type=int, default=0,
                        help='Normalization Mode', dest='mode')
    parser.add_argument('-o', '--out_dir', type=str,
                        help='Output Directory', dest='save_dir')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)
