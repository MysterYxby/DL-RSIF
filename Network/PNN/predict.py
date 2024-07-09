import torch
import torch.nn as nn
import imageio
from torch.nn import functional as F
from model_pnn import PNN
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
from torchvision import transforms
import os
import cv2 
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')


def read_h5_to_tensors(h5_file_path):
    """
    读取HDF5文件中的多个数据集，并根据通道数将其转换为PyTorch tensors。
    """
    with h5py.File(h5_file_path, 'r') as h5_file:
        # 读取数据并转换为tensor
        lms_data = torch.from_numpy(h5_file['lms'][:])
        ms_data = torch.from_numpy(h5_file['ms'][:])
        pan_data = torch.from_numpy(h5_file['pan'][:])
            
    
    # return tensor[keys[0]],tensor[keys[1]],tensor[keys[2]]
    return lms_data,ms_data,pan_data

def TensorToIMage(image_tensor):
    c,m,n = image_tensor.size(-3),image_tensor.size(-2),image_tensor.size(-1)
    image_tensor = (image_tensor+1)*127.5
    if c == 1:
        image_np = image_tensor.detach().numpy().reshape(m,n)
    else:
        image_np = np.zeros((m,n,c))
        for i in range(c):
            image_np[:,:,i] = image_tensor[i,:,:]
            pass
    image_np[np.where(image_np < 0)] = 0
    image_np[np.where(image_np > 255)] = 255
    
    return image_np.astype(np.uint8)

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


def load_pretrained_model(model_path, model):
    # 加载预训练的模型权重
    model.load_state_dict(torch.load(model_path,map_location = device))
    model.eval()  # 将模型设置为评估模式
    return model

def process_image_and_save_tif(model, input_tensor, output_path):
    with torch.no_grad():  # 不计算梯度，以加快推理速度
        # 将numpy数组转换为torch tensor，并添加batch维度
        input_tensor = input_tensor.float().div(2047)
        # 执行模型推理
        output_tensor = model(input_tensor)  # 移除batch维度
        # print(output_tensor.shape)
        # 将输出tensor转换回numpy数组
        output_image = TensorToIMage(output_tensor.squeeze(0))
        # visualizer = MultispectralImageVisualizer8(output_image, bands=[0, 2, 4], dim_cut=10)
        # visualizer.view_and_save('path_to_save_image.tif')  # 可视化并保存图像
        # visualizer.save_image(id=1)  # 保存图像
        # 使用imageio保存为TIF格式
        # imageio.imwrite(output_path, output_image)
        imageio.imwrite(output_path, RSGenerate(output_image[:, :, [4, 2, 1]], 1, 1))
        '''
        coastal blue, blue, green, yellow, red, red edge, NIR1, and NIR2).
        '''

if __name__ =='__main__':

    # 使用示例
    h5_file_path = 'D:/Pan//WorldView3/reduced_examples/test_wv3_multiExm1.h5'
    # h5_file_path = 'D:/Pan//QuickBrid/full_examples/test_qb_OrigScale_multiExm1.h5'
    output_dir  = 'output/PNN/qb'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 读取HDF5文件并转换为tensors
    lms_data, ms_data, pan_data = read_h5_to_tensors(h5_file_path)
    # print(lms_data.shape,ms_data.shape,pan_data.shape)
    # print(demo.shape)
    # 假设我们已经有了一个模型实例model和预训练权重的路径model_path
    model_path = 'pretrained-model/QB/pnn.pth'
    model = PNN(spectral_num=4)
    pretrained_model = load_pretrained_model(model_path, model)
    for i in range(lms_data.shape[0]):



        # 假设input_image是我们想要处理的图像numpy数组
        # input_tensor = torch.cat((pan_data[i].unsqueeze(0),lms_data[i].unsqueeze(0)),1)
        input_tensor = torch.cat((lms_data[i].unsqueeze(0),pan_data[i].unsqueeze(0)),1)
        # 定义输出TIF文件的路径
        
        output_tif_path = os.path.join(output_dir, f'{i+1}.tif')

        # 调用函数处理图像并保存为TIF
        process_image_and_save_tif(pretrained_model, input_tensor, output_tif_path)