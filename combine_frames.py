import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import os
import cv2
import sys
from pathlib import Path


home = str(Path.home())
base_dir = os.path.join(home, "dev/data/Demo/frames_for_demo_video")
outdir = os.path.join(base_dir, 'out')
if not os.path.exists(outdir):
    os.makedirs(outdir)

data_path_au_rmi = "au_rmi"
data_path_au_cl = "au_cl"
data_path_au_cl_tiaug = "au_cl_tiaug"
data_path_samcl = "SAM-CL"

data_path_raw_images = os.path.join(home, "dev/data/Demo/Processed/test/image")
raw_ext = '.npy'
img_ext = '.jpg'
fnames = os.listdir(os.path.join(base_dir, data_path_au_rmi))
total_frame_count = len(fnames)
# video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60, (400, 1200))

'''
Generate .mp4 from many image frames
    ffmpeg -pattern_type glob -i '*.jpg' -c:v libx264 -framerate 30 -filter:v "setpts=PTS/4" SAM-CL_Demo.mp4

Generate .gif from .mp4
    palette="/tmp/palette.png"
    filters="fps=15,scale=1200:-1:flags=lanczos"
    ffmpeg -i SAM-CL_Demo.mp4 -vf "$filters,palettegen" -y $palette
    ffmpeg -i SAM-CL_Demo.mp4 -i $palette -lavfi "$filters [x]; [x][1:v] paletteuse" -y SAM-CL_Demo.gif

see -> https://superuser.com/questions/556029/how-do-i-convert-a-video-to-gif-using-ffmpeg-with-reasonable-quality
'''

for i in range(total_frame_count):
    fn = f'{i+1:04d}'
    img = np.load(os.path.join(data_path_raw_images, fn + raw_ext))
    img_au_rmi = cv2.imread(os.path.join(base_dir, data_path_au_rmi, fn + img_ext))
    img_au_cl = cv2.imread(os.path.join(base_dir, data_path_au_cl, fn + img_ext))
    # img_au_cl_tiaug = cv2.imread(os.path.join(base_dir, data_path_au_cl_tiaug, fn + img_ext))
    img_samcl = cv2.imread(os.path.join(base_dir, data_path_samcl, fn + img_ext))
    
    img_au_rmi = cv2.cvtColor(img_au_rmi, cv2.COLOR_RGB2BGR)
    img_au_cl = cv2.cvtColor(img_au_cl, cv2.COLOR_RGB2BGR)
    # # img_au_cl_tiaug = cv2.cvtColor(img_au_cl_tiaug, cv2.COLOR_RGB2BGR)
    img_samcl = cv2.cvtColor(img_samcl, cv2.COLOR_RGB2BGR)
    
    img_au_rmi = img_au_rmi[58:426, 96:558]
    img_au_cl = img_au_cl[58:426, 96:558]
    # # img_au_cl_tiaug = img_au_cl_tiaug[58:426, 96:558]
    img_samcl = img_samcl[58:426, 96:558]

    fig = plt.figure(figsize=(12, 3.5), tight_layout=True)
    spec = gridspec.GridSpec(nrows=1, ncols=4, figure=fig)

    ax_0 = fig.add_subplot(spec[0, 0])
    ax_0.imshow(img, cmap='gray')
    ax_0.set_title('Original Thermal Matrix', fontsize=12)
    ax_0.axis('off')

    ax_1 = fig.add_subplot(spec[0, 1])
    ax_1.imshow(img_au_rmi)
    ax_1.set_title('Attention-UNET\n+RMI Loss', fontsize=12)
    ax_1.axis('off')

    ax_2 = fig.add_subplot(spec[0, 2])
    ax_2.imshow(img_au_cl)
    ax_2.set_title('Attention-UNET\n+Contrastive Learning', fontsize=12)
    ax_2.axis('off')

    # ax_3 = fig.add_subplot(spec[0, 3])
    # ax_3.imshow(img_au_cl_tiaug)
    # ax_3.set_title('Attention-UNET\n+Contrastive Learning' + r"$\bf{ + TiAug}$", fontsize=12)
    # ax_3.axis('off')

    ax_4 = fig.add_subplot(spec[0, 3])
    ax_4.imshow(img_samcl)
    # ax_4.set_title('Attention-UNET\n' + r"$\bf{+ SAM}$" + "-" + r"$\bf{CL + TiAug}$", fontsize=12)
    ax_4.set_title('Attention-UNET\n' + r"$\bf{+ SAM}$" + "-" + r"$\bf{CL}$" + ' ' + r"$\bf{Framework}$", fontsize=12)
    ax_4.axis('off')

    # plt.show()
    # break
    plt.savefig(os.path.join(outdir, fn+'.jpg'))

    # put pixel buffer in numpy array
    # canvas = FigureCanvas(fig)
    # canvas.draw()
    # mat = np.array(canvas.renderer._renderer)
    # mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)

    # # write frame to video
    # video.write(mat)
    
    sys.stdout.write("Processing: " + str(i) + " of " + str(total_frame_count) + "\r")
    sys.stdout.flush()

    plt.close()
    plt.cla()


# # close video writer
# cv2.destroyAllWindows()
# video.release()
