import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

import pandas as pd


def main(args):
   
    base_dir = args.base_dir    
    data_dirs = []
    base_dir_list = os.listdir(base_dir)

    for ff in range(len(base_dir_list)):
        dir_name = base_dir_list[ff]
        data_dirs.append(os.path.join(base_dir, dir_name))

    data_dict = {}
    data_dict['min'] = {}
    data_dict['avg'] = {}
    data_dict['max'] = {}
    data_dict['std'] = {}

    # fig, ax = plt.subplots(figsize=(6, 4))
    # colors = {'short': 'MediumVioletRed', 'long': 'Navy'}

    for i, fn in enumerate(data_dirs):
        if data_dirs[i] not in data_dict['min']:
            data_dict['min'][data_dirs[i]] = np.load(os.path.join(base_dir, data_dirs[i], 'min_array.npy'))
            data_dict['avg'][data_dirs[i]] = np.load(os.path.join(base_dir, data_dirs[i], 'avg_array.npy'))
            data_dict['max'][data_dirs[i]] = np.load(os.path.join(base_dir, data_dirs[i], 'max_array.npy'))
            data_dict['std'][data_dirs[i]] = np.load(os.path.join(base_dir, data_dirs[i], 'std_array.npy'))


    df = pd.DataFrame.from_dict(data_dict, orient='index')

    df.plot.scatter(data_dict['min'].keys(), data_dict['min'].values())

    # ax.set(xlabel='Waiting', ylabel='Duration')
    # fig.suptitle('Waiting vs Duration')
    # plt.show()

    # plt.show()
    plt.savefig(os.path.join(base_dir, 'box_plot.jpg'), bbox_inches=0)


def get_args():
    parser = argparse.ArgumentParser(description='Generate the box plot for temperature ranges in multiple datasets',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--analysis_dir', type=str,
                        help='Analysis Directory', dest='base_dir')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)
