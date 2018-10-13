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


def get_test_image_list():
    bg_by_user_test_list = file_name("../data/bg_by_user/test")
    image_list = []
    for file in bg_by_user_test_list:
        sub_image_list = []
        sub_image_list.append(file)
        temp_file = file.replace("../data/bg_by_user/test", "../data/foreground/test")
        sub_image_list.append(temp_file)
        temp_file = file.replace("../data/bg_by_user/test", "../data/inner_mask/test")
        sub_image_list.append(temp_file)
        image_list.append(sub_image_list)
    return image_list


def get_train_image_list():
    bg_by_user_train_list = file_name("../data/bg_by_user/train")
    image_list = []
    for file in bg_by_user_train_list:
        sub_image_list = []
        sub_image_list.append(file)
        temp_file = file.replace("../data/bg_by_user/train", "../data/foreground/train")
        sub_image_list.append(temp_file)
        temp_file = file.replace("../data/bg_by_user/train", "../data/inner_mask/train")
        sub_image_list.append(temp_file)
        image_list.append(sub_image_list)
    return image_list

l = get_train_image_list()
print(l)