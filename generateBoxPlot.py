from turtle import color
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

    data_dict_min = {}
    data_dict_min['x'] = []
    data_dict_min['y'] = []

    data_dict_avg = {}
    data_dict_avg['x'] = []
    data_dict_avg['y'] = []
    
    data_dict_max = {}
    data_dict_max['x'] = []
    data_dict_max['y'] = []

    data_dict_std = {}
    data_dict_std['x'] = []
    data_dict_std['y'] = []
    
    data_dict_fg_avg = {}
    data_dict_fg_avg['x'] = []
    data_dict_fg_avg['y'] = []

    data_dict_bg_avg = {}
    data_dict_bg_avg['x'] = []
    data_dict_bg_avg['y'] = []

    data_dict_fg_max_diff = {}
    data_dict_fg_max_diff['x'] = []
    data_dict_fg_max_diff['y'] = []

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

        fs_fg_max_diff = os.path.join(data_dirs[i], 'fg_max_diff_array.npy')
        if os.path.exists(fs_fg_max_diff):
            fg_max_diff_array = np.load(fs_fg_max_diff)

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
        elif np.all(fg_max_diff_array) != None:
            instance_count = len(fg_max_diff_array)

        for j in range(instance_count):
            if np.all(min_array) != None:
                data_dict_min['x'].append(dir_names[i])
                data_dict_min['y'].append(min_array[j])
            if np.all(avg_array) != None:
                data_dict_avg['x'].append(dir_names[i])
                data_dict_avg['y'].append(avg_array[j])
            if np.all(max_array) != None:
                data_dict_max['x'].append(dir_names[i])
                data_dict_max['y'].append(max_array[j])
            if np.all(std_array) != None:
                data_dict_std['x'].append(dir_names[i])
                data_dict_std['y'].append(std_array[j])
            if np.all(fg_avg_array) != None:
                data_dict_fg_avg['x'].append(dir_names[i])
                data_dict_fg_avg['y'].append(fg_avg_array[j])
            if np.all(bg_avg_array) != None:
                data_dict_bg_avg['x'].append(dir_names[i])
                data_dict_bg_avg['y'].append(bg_avg_array[j])
            if np.all(fg_max_diff_array) != None:
                data_dict_fg_max_diff['x'].append(dir_names[i])
                data_dict_fg_max_diff['y'].append(fg_max_diff_array[j])

    # df_min = pd.DataFrame.from_dict(data_dict_min)
    # df_max = pd.DataFrame.from_dict(data_dict_max)
    df_avg = pd.DataFrame.from_dict(data_dict_avg)
    df_std = pd.DataFrame.from_dict(data_dict_std)
    df_fg_avg = pd.DataFrame.from_dict(data_dict_fg_avg)
    df_bg_avg = pd.DataFrame.from_dict(data_dict_bg_avg)

    data_dict_fgbg_diff = {}
    data_dict_fgbg_diff['x'] = []
    data_dict_fgbg_diff['y'] = []
    for i in range(len(data_dict_fg_avg['y'])):
        data_dict_fgbg_diff['x'].append(data_dict_fg_avg['x'][i])
        data_dict_fgbg_diff['y'].append(data_dict_fg_avg['y'][i] - data_dict_bg_avg['y'][i] - 0.75)
    df_fgbg_diff = pd.DataFrame.from_dict(data_dict_fgbg_diff)
    # df_fgbg_diff = pd.DataFrame.from_dict(data_dict_fg_max_diff)

    if len(data_dict_avg['y']) > 0:
        sns.boxplot(x='x', y='y', data=df_avg, whis=4.0)
        plt.xlabel('Different Datasets')
        plt.ylabel('Average Value')
        plt.title('Box Plot Analysis')
        plt.savefig(os.path.join(base_dir, 'boxplot_avg.jpg'), bbox_inches=0)
        plt.close()

    if len(data_dict_std['y']) > 0:
        sns.boxplot(x='x', y='y', data=df_std, whis=4.0)
        plt.xlabel('Different Datasets')
        plt.ylabel('Standard Deviation')
        plt.title('Box Plot Analysis')
        plt.savefig(os.path.join(base_dir, 'boxplot_std.jpg'), bbox_inches=0)
        plt.close()
    
    if len(data_dict_fg_avg['y']) > 0:
        sns.boxplot(x='x', y='y', data=df_fg_avg, whis=4.0)
        if len(data_dict_bg_avg['y']) > 0:
            sns.boxplot(x='x', y='y', data=df_bg_avg, whis=4.0)
        plt.xlabel('Average Foreground and Background Temperature')
        plt.ylabel('Average Value')
        plt.title('Box Plot Analysis')
        plt.savefig(os.path.join(base_dir, 'boxplot_foreground_background_avg.jpg'), bbox_inches=0)
        plt.close()

    if len(data_dict_fg_max_diff['y']) > 0:
        sns.boxplot(x='x', y='y', data=df_fgbg_diff, whis=4.0)
        plt.xlabel('Use of Augmentation Technique - TiAug')
        plt.ylabel('Difference in Foreground-Background Temperature')
        plt.title('Box Plot Analysis')
        plt.savefig(os.path.join(base_dir, 'boxplot_foreground_background_diff.jpg'), bbox_inches=0)
        plt.close()

    if len(data_dict_avg['y']) > 0:
        sns.scatterplot(x='x', y='y', data=df_avg)
        plt.xlabel('Different Datasets')
        plt.ylabel('Average Value')
        plt.title('Scatter Plot Analysis')
        plt.savefig(os.path.join(base_dir, 'scatterplot_avg.jpg'), bbox_inches=0)
        plt.close()

    if len(data_dict_std['y']) > 0:
        sns.scatterplot(x='x', y='y', data=df_std)
        plt.xlabel('Different Datasets')
        plt.ylabel('Standard Deviation')
        plt.title('Scatter Plot Analysis')
        plt.savefig(os.path.join(base_dir, 'scatterplot_std.jpg'), bbox_inches=0)
        plt.close()

    if len(data_dict_fg_avg['y']) > 0:
        sns.scatterplot(x='x', y='y', data=df_fg_avg)
        if len(data_dict_bg_avg['y']) > 0:
            sns.scatterplot(x='x', y='y', data=df_bg_avg)
        plt.xlabel('Average Foreground and Background Temperature')
        plt.ylabel('Average Value')
        plt.title('Scatter Plot Analysis')
        plt.savefig(os.path.join(base_dir, 'scatterplot_foreground_background_avg.jpg'), bbox_inches=0)
        plt.close()

    if len(data_dict_fg_max_diff['y']) > 0:
        sns.scatterplot(x='x', y='y', data=df_fgbg_diff)
        plt.xlabel('Use of Augmentation Technique - TiAug')
        plt.ylabel('Difference in Foreground-Background Temperature')
        plt.title('Box Plot Analysis')
        plt.savefig(os.path.join(base_dir, 'scatterplot_foreground_background_diff.jpg'), bbox_inches=0)
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
