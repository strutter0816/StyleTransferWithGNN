import os

import cv2

VIDEO_NAME = "../video.mp4"
FRAME_SAVE_PATH = "./frames/"
FRAME_CONTENT_FOLDER = "content_folder/"
FRAME_BASE_FILE_NAME = "frame"
STYLE_FRAME_BASE_FILE_NAME = "style1_stylized_frame"
FRAME_BASE_FILE_TYPE = ".jpg"
STYLE_FRAME_SAVE_PATH = "../Frames_output/images/"
STYLE_VIDEO_NAME = "test.mp4"
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
    count = 1
    success = True
    while success:
        cv2.imwrite("{}{}{}{}".format(FRAME_SAVE_PATH + FRAME_CONTENT_FOLDER, FRAME_BASE_FILE_NAME, count,
                                      FRAME_BASE_FILE_TYPE), image)
        success, image = vidcap.read()
        count += 1
    print("Done extracting all frames")
def makeVideo(frames_path, save_name, fps, height, width):
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
if __name__ == '__main__':
    H,W,fps =  getInfo(VIDEO_NAME)
    print("Height:{} Width:{} fps:{}".format(H,W,fps))
    # getFrames(VIDEO_NAME,FRAME_SAVE_PATH,FRAME_CONTENT_FOLDER,FRAME_BASE_FILE_NAME,FRAME_BASE_FILE_TYPE)
#     makeVideo(STYLE_FRAME_SAVE_PATH, STYLE_VIDEO_NAME, 30, 512, 512)