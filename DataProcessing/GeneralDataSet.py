import os
import time
from random import sample

import numpy as np
import pyopencl as cl
from PIL import Image
from tqdm import tqdm


def timeit(func):
    def wrapper():
        start = time.clock()
        func()
        end = time.clock()
        print('time elasped:', end - start)

    return wrapper


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


@timeit
def generate_image_data_using_pillow():
    if not os.path.exists('../data/expand_data/test/'):
        os.makedirs('../data/expand_data/test/')
    if not os.path.exists('../data/expand_data/train/'):
        os.makedirs('../data/expand_data/train/')

    imgs = get_image_list_from_class('train') + get_image_list_from_class('test')
    for index in tqdm(range(imgs.__len__())):
        img_src_paths = imgs[index]

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

        new_color_list = [(153, 217, 234), (181, 230, 29)] + [get_random_two_color() for i in range(3)]

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


@timeit
def generate_image_data_using_opencl():

    # import os
    # os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

    if not os.path.exists('../data/expand_data/test/'):
        os.makedirs('../data/expand_data/test/')
    if not os.path.exists('../data/expand_data/train/'):
        os.makedirs('../data/expand_data/train/')

    # 初始化OpenCL 环境
    CL_SOURCE = '''//CL//
    __kernel void convert(
        read_only image2d_t usr,
        read_only image2d_t ori,
        write_only image2d_t dest,
        __global uchar* mas,
        const uint4 new_sky_color,
        const uint4 new_ground_color
    )
    {
        const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
        int2 pos = (int2)(get_global_id(0), get_global_id(1));

        uchar mask = mas[pos.x+pos.y*get_global_size(0)];

        uint4 usr_pix = read_imageui(usr,sampler,pos);
        uint4 ori_pix = read_imageui(ori,sampler,pos);
        uint4 out_pix = ori_pix;

        if( mask == 255 ){
            if(all(usr_pix == (uint4)(153, 217, 234, 255))){
                out_pix = new_sky_color;
            }
            else if(all(usr_pix == (uint4)(181, 230, 29, 255))) 
                 out_pix = new_ground_color;
        }

        write_imageui(dest, pos, out_pix);
    }
    '''

    ctx = cl.Context(dev_type=cl.device_type.GPU)
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, CL_SOURCE).build()
    mf = cl.mem_flags

    imgs = get_image_list_from_class('train') + get_image_list_from_class('test')
    for index in tqdm(range(imgs.__len__())):
        img_paths = imgs[index]
        # 读取图片
        usr_img = Image.open(img_paths[0]).convert('RGBA')
        ori_img = Image.open(img_paths[1]).convert('RGBA')
        msk_img = Image.open(img_paths[2]).convert('L')

        # 生成 GL BUFFER 对象
        usr = np.array(usr_img)
        ori = np.array(ori_img)
        msk = np.array(msk_img).reshape(-1).astype(np.uint8)

        h, w, c = usr.shape
        usr_buf = cl.image_from_array(ctx, usr, c)
        ori_buf = cl.image_from_array(ctx, ori, c)
        msk_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=msk)
        fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
        dest_buf = cl.Image(ctx, mf.WRITE_ONLY, fmt, shape=(w, h))

        dest = np.empty_like(usr_img)
        save_path = img_paths[0].replace("bg_by_user", "expand_data")[:-4]

        for i in range(4):
            # 生成新的随机颜色
            new_sky_color, new_ground_color = [(153, 217, 234), (181, 230, 29)] if i == 0 else get_random_two_color()
            new_sky_color = np.array(new_sky_color + (255,), dtype=np.uint32)
            new_ground_color = np.array(new_ground_color + (255,), dtype=np.uint32)

            # 调用核函数
            prg.convert(queue, (w, h), None, usr_buf, ori_buf, dest_buf, msk_buf, new_sky_color, new_ground_color)

            # 从GPU中复制数据
            cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(w, h))
            dest_img = Image.fromarray(dest)
            dest_img.save("%s_%d.png" % (save_path, i + 1))


# generate_image_data_using_pillow() 这个很慢
generate_image_data_using_opencl()
