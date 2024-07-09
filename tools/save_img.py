import os
import numpy as np
import cv2
from scipy.io import loadmat
import imageio


def RSGenerate(image, percent, colorization=True):
    #   RSGenerate(image,percent,colorization)
    #                               --Use to correct the color
    # image should be R G B format with three channels
    # percent is the ratio when restore whose range is [0,100]
    # colorization is True
    m, n, c = image.shape
    # print(np.max(image))
    image_normalize = image / np.max(image)
    image_generate = np.zeros(list(image_normalize.shape))
    if colorization:
        # Multi-channel Image R,G,B
        for i in range(c):
            image_slice = image_normalize[:, :, i]
            pixelset = np.sort(image_slice.reshape([m * n]))
            maximum = pixelset[np.floor(m * n * (1 - percent / 100)).astype(np.int32)]
            minimum = pixelset[np.ceil(m * n * percent / 100).astype(np.int32)]
            image_generate[:, :, i] = (image_slice - minimum) / (maximum - minimum + 1e-9)
            pass
        image_generate[np.where(image_generate < 0)] = 0
        image_generate[np.where(image_generate > 1)] = 1
        image_generate = cv2.normalize(image_generate, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    return image_generate.astype(np.uint16)

def save_image(ms_image, save_dir, file_name, bands, flag_cut_bounds, dim_cut, ratio):
    # 保存图像函数，适用于四波段或八波段图像
    #RGB
    if flag_cut_bounds:
        w, h,c = ms_image.shape
        ms_image = ms_image[round(dim_cut/ratio):-round(dim_cut/ratio), 
                             round(dim_cut/ratio):-round(dim_cut/ratio), :]
    selected_bands = ms_image[:, :, bands]
    Rs_img = RSGenerate(selected_bands,1,1)
    if flag_cut_bounds:
        new_size = (w,h)
        Rs_img = cv2.resize(Rs_img, new_size)
    imageio.imwrite(os.path.join(save_dir, file_name), Rs_img)


def save_images(method, dataset, set_type):
    base_dir = 'output/'
    fusion_dir = os.path.join(base_dir, method, dataset, set_type, 'mat_img')
    save_dir = os.path.join(base_dir, method, dataset, set_type, 'tif_rs_rgb')
    os.makedirs(save_dir, exist_ok=True)

    files_fusion = [f for f in os.listdir(fusion_dir) if f.endswith('.mat')]
    for file in files_fusion:
        file_path_fusion = os.path.join(fusion_dir, file)
        data = loadmat(file_path_fusion, mat_dtype=True)
        ms_image = data['ms_image']  # 确保这里正确地引用了图像数据变量

        file_name = os.path.splitext(os.path.basename(file))[0] + '.tif'
        num_bands = ms_image.shape[2]  # 获取图像的波段数
        if num_bands == 4:  
            bands = [2, 1, 0]
        else:
            bands = [4, 2, 1]
        save_image(ms_image, save_dir, file_name, bands, 1, 21, 4)

# 示例用法
dataset_names = ['gf2', 'qb', 'wv2', 'wv3']
set_type_names = ['reduced', 'full']
method = 'BDPN'

for dataset in dataset_names:
    for set_type in set_type_names:
        save_images(method, dataset, set_type)