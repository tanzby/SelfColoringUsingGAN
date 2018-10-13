import cv2 as opencv
import os


# 其中os.path.splitext()函数将路径拆分为文件名+扩展名
def file_name(file_dir):
    file_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                file_list.append(os.path.join(root, file))
    return file_list

