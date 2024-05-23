import os

import cv2
FRAME_SAVE_PATH = "./content/1/"
FRAME_CONTENT_FOLDER = "content_folder/"
FRAME_BASE_FILE_NAME = "frame"
STYLE_FRAME_BASE_FILE_NAME = "style1_stylized_frame"
FRAME_BASE_FILE_TYPE = ".jpg"
# STYLE_FRAME_SAVE_PATH = "../Frames_output/images/"
STYLE_VIDEO_NAME = "test.mp4"
def getFrames(video_path):
    """
    Extracts the frames of a video
    and saves in specified path
    """
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 1
    success = True
    while success:
        cv2.imwrite("{}{}{}{}".format(FRAME_SAVE_PATH , FRAME_BASE_FILE_NAME, count,
                                      FRAME_BASE_FILE_TYPE), image)
        success, image = vidcap.read()
        count += 1