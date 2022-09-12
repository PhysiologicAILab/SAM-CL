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
        min_array = np.load(os.path.join(data_dirs[i], 'min_array.npy'))
        avg_array = np.load(os.path.join(data_dirs[i], 'avg_array.npy'))
        max_array = np.load(os.path.join(data_dirs[i], 'max_array.npy'))
        std_array = np.load(os.path.join(data_dirs[i], 'std_array.npy'))
        fg_avg_array = np.load(os.path.join(data_dirs[i], 'fg_avg_array.npy'))
        bg_avg_array = np.load(os.path.join(data_dirs[i], 'bg_avg_array.npy'))

        for j in range(len(min_array)):
            data_dict['com_x'].append(dir_names[i])
            data_dict['min_y'].append(min_array[j])
            data_dict['avg_y'].append(avg_array[j])
            data_dict['max_y'].append(max_array[j])
            data_dict['std_y'].append(std_array[j])
            data_dict['fg_avg_y'].append(fg_avg_array[j])
            data_dict['bg_avg_y'].append(bg_avg_array[j])


    df = pd.DataFrame.from_dict(data_dict)

    sns.boxplot(x='com_x', y='avg_y', data=df)
    plt.xlabel('Different Datasets')
    plt.ylabel('Average Value')
    plt.title('Box Plot Analysis')
    plt.savefig(os.path.join(base_dir, 'boxplot_avg.jpg'), bbox_inches=0)
    plt.close()

    sns.boxplot(x='com_x', y='std_y', data=df)
    plt.xlabel('Different Datasets')
    plt.ylabel('Standard Deviation')
    plt.title('Box Plot Analysis')
    plt.savefig(os.path.join(base_dir, 'boxplot_std.jpg'), bbox_inches=0)
    plt.close()
    
    sns.boxplot(x='com_x', y='fg_avg_y', data=df)
    sns.boxplot(x='com_x', y='bg_avg_y', data=df)
    plt.xlabel('Average Foreground and Background Temperature')
    plt.ylabel('Average Value')
    plt.title('Box Plot Analysis')
    plt.savefig(os.path.join(base_dir, 'boxplot_foreground_background_avg.jpg'), bbox_inches=0)
    plt.close()

    sns.scatterplot(x='com_x', y='avg_y', data=df)
    plt.xlabel('Different Datasets')
    plt.ylabel('Average Value')
    plt.title('Scatter Plot Analysis')
    plt.savefig(os.path.join(base_dir, 'scatterplot_avg.jpg'), bbox_inches=0)
    plt.close()

    sns.scatterplot(x='com_x', y='std_y', data=df)
    plt.xlabel('Different Datasets')
    plt.ylabel('Standard Deviation')
    plt.title('Scatter Plot Analysis')
    plt.savefig(os.path.join(base_dir, 'scatterplot_std.jpg'), bbox_inches=0)
    plt.close()

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
