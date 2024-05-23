import os

import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import pdb


def save_all_image(output_list, style_list, filename, row=8):
    if row>len(output_list):
        row = len(output_list)
    col=int(len(output_list)/row*2)
    plt.figure(figsize=(row*16, col*16))
    
    for i in range(int(col/2)):
        for j in range(row):
            ax = plt.subplot(col, row, i*2*row+j+1)
            ax.imshow(style_list[i*row+j].squeeze(0).permute(1,2,0))
            ax.axis('off')
            ax = plt.subplot(col, row, (i*2+1)*row+j+1)
            ax.imshow(output_list[i*row+j].squeeze(0).permute(1,2,0))
            ax.axis('off')
    plt.savefig(filename)
    
def save_all_image_v1(output_list, style_list, filename, labels=None):
    row = len(output_list)+1
    col=len(style_list)
    plt.figure(figsize=(row*8, col*8))
    
    for i in range(col):
        ax = plt.subplot(col, row, i*row+1)
        _,_,h,w = style_list[i].shape
        if h>w:
            ax.imshow(style_list[i].squeeze(0).permute(1,2,0)[int((h-w)/2):int((h+w)/2),:])
        else:
            ax.imshow(style_list[i].squeeze(0).permute(1,2,0)[:,int((w-h)/2):int((w+h)/2)])
        ax.axis('off')
        for j in range(1, row):
            ax = plt.subplot(col, row, i*row+j+1)
            _,_,h,w = output_list[j-1][i].shape
            if h>w:
                ax.imshow(output_list[j-1][i].squeeze(0).permute(1,2,0)[int((h-w)/2):int((h+w)/2),:])
            else:
                ax.imshow(output_list[j-1][i].squeeze(0).permute(1,2,0)[:,int((w-h)/2):int((w+h)/2)])
            ax.axis('off')
    plt.savefig(filename, bbox_inches = 'tight', pad_inches=0.2)
    
#     for video style transfer

# VIDEO_NAME = "video.mp4"
# FRAME_SAVE_PATH = "./frames/"
# FRAME_CONTENT_FOLDER = "./videotest/frames/content_folder/"
# FRAME_BASE_FILE_NAME = "frame"
# STYLE_FRAME_BASE_FILE_NAME = "stylized_frame"
# FRAME_BASE_FILE_TYPE = ".jpg"
# STYLE_FRAME_SAVE_PATH = "./Frames_output/stylized_frames/"
# STYLE_VIDEO_NAME = "test.mp4"

def getInfo(video_path):
    """
    Extracts the height, width,
    and fps of a video
    """
    vidcap = cv2.VideoCapture(video_path)
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    # print(vidcap+"--"+width+"--"+height+"--"+fps)
    return height, width, fps


def getFrames(video_path, FRAME_SAVE_PATH, FRAME_CONTENT_FOLDER, FRAME_BASE_FILE_NAME,
              FRAME_BASE_FILE_TYPE):
    """
    Extracts the frames of a video
    and saves in specified path
    """
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
#     print("get frames check",success)
    count = 1
    success = True
    while success:
        cv2.imwrite("{}{}{}{}".format(FRAME_CONTENT_FOLDER, FRAME_BASE_FILE_NAME, count,
                                      FRAME_BASE_FILE_TYPE), image)
        success, image = vidcap.read()
#         print("get frames check",success)
        count += 1
#     print("check count",count)
    print("Done extracting all frames ")
def makeVideo(frames_path, save_name, fps, height, width,STYLE_FRAME_BASE_FILE_NAME,FRAME_BASE_FILE_TYPE):
    # Extract image paths. Natural sorting of directory list. Python does not have a native support for natural sorting :(
    base_name_len = len(STYLE_FRAME_BASE_FILE_NAME)
    filetype_len = len(FRAME_BASE_FILE_TYPE)
    images = [img for img in sorted(os.listdir(frames_path), key=lambda x : int(x[base_name_len:-filetype_len])) if img.endswith(".jpg")]

    # Define the codec and create VideoWrite object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vout = cv2.VideoWriter(save_name, fourcc, fps, (width,height))

    # Write the video
    for image_name in images:
        vout.write(cv2.imread(os.path.join(frames_path, image_name)))

    print("Done writing video")
