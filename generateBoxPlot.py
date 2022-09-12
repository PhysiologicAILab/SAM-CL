import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def main(args):
   
    base_dir = args.base_dir    
    data_dirs = []
    dir_names = []
    base_dir_list = os.listdir(base_dir)

    for ff in range(len(base_dir_list)):
        dir_name = base_dir_list[ff]
        if os.path.isdir(os.path.join(base_dir, dir_name)):
            dir_names.append(dir_name)
            data_dirs.append(os.path.join(base_dir, dir_name))

    data_dict = {}
    data_dict['com_x'] = []
    data_dict['min_y'] = []
    data_dict['avg_y'] = []
    data_dict['max_y'] = []
    data_dict['std_y'] = []
    data_dict['fg_avg_y'] = []
    data_dict['bg_avg_y'] = []

    for i, fn in enumerate(data_dirs):
        min_array = None
        avg_array = None
        max_array = None
        std_array = None
        fg_avg_array = None
        bg_avg_array = None

        fs_min = os.path.join(data_dirs[i], 'min_array.npy')
        if os.path.exists(fs_min):
            min_array = np.load(fs_min)

        fs_avg = os.path.join(data_dirs[i], 'avg_array.npy')
        if os.path.exists(fs_avg):
            avg_array = np.load(fs_avg)

        fs_max_ = os.path.join(data_dirs[i], 'max_array.npy')
        if os.path.exists(fs_max_):
            max_array = np.load(fs_max_)

        fs_std = os.path.join(data_dirs[i], 'std_array.npy')
        if os.path.exists(fs_std):
            std_array = np.load(fs_std)

        fs_fg_avg = os.path.join(data_dirs[i], 'fg_avg_array.npy')
        if os.path.exists(fs_fg_avg):
            fg_avg_array = np.load(fs_fg_avg)

        fs_bg_avg = os.path.join(data_dirs[i], 'bg_avg_array.npy')
        if os.path.exists(fs_bg_avg):
            bg_avg_array = np.load(fs_bg_avg)

        instance_count = 0
        if np.all(min_array) != None:
            instance_count = len(min_array)
        elif np.all(avg_array) != None:
            instance_count = len(avg_array)
        elif np.all(std_array) != None:
            instance_count = len(std_array)
        elif np.all(max_array) != None:
            instance_count = len(max_array)
        elif np.all(fg_avg_array) != None:
            instance_count = len(fg_avg_array)
        elif np.all(bg_avg_array) != None:
            instance_count = len(bg_avg_array)

        for j in range(instance_count):
            data_dict['com_x'].append(dir_names[i])
            if np.all(min_array) != None:
                data_dict['min_y'].append(min_array[j])
            if np.all(avg_array) != None:
                data_dict['avg_y'].append(avg_array[j])
            if np.all(max_array) != None:
                data_dict['max_y'].append(max_array[j])
            if np.all(std_array) != None:
                data_dict['std_y'].append(std_array[j])
            if np.all(fg_avg_array) != None:
                data_dict['fg_avg_y'].append(fg_avg_array[j])
            if np.all(bg_avg_array) != None:
                data_dict['bg_avg_y'].append(bg_avg_array[j])


    df = pd.DataFrame.from_dict(data_dict)

    if len(data_dict['avg_y']) > 0:
        sns.boxplot(x='com_x', y='avg_y', data=df)
        plt.xlabel('Different Datasets')
        plt.ylabel('Average Value')
        plt.title('Box Plot Analysis')
        plt.savefig(os.path.join(base_dir, 'boxplot_avg.jpg'), bbox_inches=0)
        plt.close()


    if len(data_dict['std_y']) > 0:
        sns.boxplot(x='com_x', y='std_y', data=df)
        plt.xlabel('Different Datasets')
        plt.ylabel('Standard Deviation')
        plt.title('Box Plot Analysis')
        plt.savefig(os.path.join(base_dir, 'boxplot_std.jpg'), bbox_inches=0)
        plt.close()
    
    if len(data_dict['fg_avg_y']) > 0:
        sns.boxplot(x='com_x', y='fg_avg_y', data=df)
        sns.boxplot(x='com_x', y='bg_avg_y', data=df)
        plt.xlabel('Average Foreground and Background Temperature')
        plt.ylabel('Average Value')
        plt.title('Box Plot Analysis')
        plt.savefig(os.path.join(base_dir, 'boxplot_foreground_background_avg.jpg'), bbox_inches=0)
        plt.close()

    if len(data_dict['avg_y']) > 0:
        sns.scatterplot(x='com_x', y='avg_y', data=df)
        plt.xlabel('Different Datasets')
        plt.ylabel('Average Value')
        plt.title('Scatter Plot Analysis')
        plt.savefig(os.path.join(base_dir, 'scatterplot_avg.jpg'), bbox_inches=0)
        plt.close()

    if len(data_dict['std_y']) > 0:
        sns.scatterplot(x='com_x', y='std_y', data=df)
        plt.xlabel('Different Datasets')
        plt.ylabel('Standard Deviation')
        plt.title('Scatter Plot Analysis')
        plt.savefig(os.path.join(base_dir, 'scatterplot_std.jpg'), bbox_inches=0)
        plt.close()

    if len(data_dict['fg_avg_y']) > 0:
        sns.scatterplot(x='com_x', y='fg_avg_y', data=df)
        sns.scatterplot(x='com_x', y='bg_avg_y', data=df)
        plt.xlabel('Average Foreground and Background Temperature')
        plt.ylabel('Average Value')
        plt.title('Scatter Plot Analysis')
        plt.savefig(os.path.join(base_dir, 'scatterplot_foreground_background_avg.jpg'), bbox_inches=0)
    plt.close()

    # plt.show()


def get_args():
    parser = argparse.ArgumentParser(description='Generate the box plot for temperature ranges in multiple datasets',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--analysis_dir', type=str,
                        help='Analysis Directory', dest='base_dir')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)
