import os
from random import sample

from PIL import Image


def get_files_full_path(file_dir):
    file_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                file_list.append(os.path.join(root, file))
    return file_list


def get_image_list_from_class(class_name):
    bg_by_user_list = get_files_full_path("../data/bg_by_user/" + class_name)
    image_list = []
    for file in bg_by_user_list:
        sub_image_list = [
            file,
            file.replace("bg_by_user", "foreground"),
            file.replace("bg_by_user", "inner_mask")
        ]
        image_list.append(sub_image_list)
    return image_list


def get_random_two_color():
    color_list = [
        (153, 217, 234), (181, 230, 29), (128, 255, 215),
        (237, 28, 36), (255, 127, 39), (255, 242, 0),
        (185, 122, 87), (163, 73, 164), (255, 174, 201),
        (30, 30, 30), (127, 127, 127)
    ]
    return sample(color_list, k=2)


def generate_image_data_using_pillow():
    if not os.path.exists('../data/expand_data/test/'):
        os.makedirs('../data/expand_data/test/')
    if not os.path.exists('../data/expand_data/train/'):
        os.makedirs('../data/expand_data/train/')

    imgs = get_image_list_from_class('train') + get_image_list_from_class('test')
    for index, img_src_paths in zip(range(imgs.__len__()), imgs):
        print("%d/%d" % (index + 1, imgs.__len__()))

        img_usr = Image.open(img_src_paths[0]).convert("RGB")
        img_mas = Image.open(img_src_paths[2]).convert("L")
        img_usr_pixdata = img_usr.load()
        img_mas_pixdata = img_mas.load()

        img_oris = [
            Image.open(img_src_paths[1]).convert("RGB"),
            Image.open(img_src_paths[1]).convert("RGB"),
            Image.open(img_src_paths[1]).convert("RGB"),
        ]

        img_ori_pixdatas = [img.load() for img in img_oris]

        new_color_list = [get_random_two_color() for i in range(3)]

        for y in range(img_mas.size[1]):
            for x in range(img_mas.size[0]):
                if img_mas_pixdata[x, y] == 255:
                    usr_pix = img_usr_pixdata[x, y]
                    for img_ori_pixdata, new_color in zip(img_ori_pixdatas, new_color_list):
                        if usr_pix == (153, 217, 234):
                            img_ori_pixdata[x, y] = new_color[0]
                        elif usr_pix == (181, 230, 29):
                            img_ori_pixdata[x, y] = new_color[1]

        save_path = img_src_paths[0].replace("bg_by_user", "expand_data")[:-4]
        for i in range(1, 4):
            img_oris[i - 1].save("%s_%d.png" % (save_path, i))


def generate_image_data_using_opencl():
    pass


generate_image_data_using_pillow()
