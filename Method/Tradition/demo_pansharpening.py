# -*- coding: utf-8 -*-
"""
License: HEU
@author: Xu
E-mail: cloudxu08@outlook.com
"""

import numpy as np
import cv2
import os
import scipy.io as sio
import torch
import h5py
from scipy.io import savemat

#方法
from EXP import EXP
from Brovey import Brovey
from PCA import PCA
from IHS import IHS
from SFIM import SFIM
from GS import GS
from Wavelet import Wavelet
from MTF_GLP import MTF_GLP
from MTF_GLP_HPM import MTF_GLP_HPM
from GSA import GSA
from CNMF import CNMF
from GFPCA import GFPCA
import imageio

class ImageTest:
    def __init__(self, Method,dataset_name):
        self.dataset_name = dataset_name
        self.Method = Method
        self.dataset_dict = {'qb':'QuickBird','gf2':'Gaofen2','wv2':'WorldView2','wv3':'WorldView3'}
        self.save_dir = os.path.join('output',self.Method,self.dataset_name)
        self.reduced_path = os.path.join('D:/Pan',self.dataset_dict[self.dataset_name],'reduced_examples/','test_' +self.dataset_name + '_multiExm1.h5')
        self.full_path = os.path.join('D:/Pan',self.dataset_dict[self.dataset_name],'full_examples/','test_' +self.dataset_name + '_OrigScale_multiExm1.h5')  
        self.model_dict= {
            'Bicubic':EXP,'Brovey':Brovey,'PCA':PCA,'IHS':IHS,'SFIM':SFIM,
            'GS':GS,'Wavelet':Wavelet,'MTF_GLP':MTF_GLP,'MTF_GLP_HPM':MTF_GLP_HPM,
            'GSA':GSA,'CNMF':CNMF,'GFPCA':GFPCA}
        self.model = self.model_dict[Method]
    def read_h5_to_tensors(self, h5_file_path):
        """
        读取HDF5文件中的多个数据集，并转换为Numpy。
        """
        '''
        lms: 上采样多光谱图像   256x256
        ms:  多光谱图像         64x64
        pan: pan图像           256x256
        '''
        with h5py.File(h5_file_path, 'r') as h5_file:
            lms_data = h5_file['lms'][:]
            ms_data = h5_file['ms'][:]
            pan_data = h5_file['pan'][:]
        return lms_data, ms_data, pan_data
    
    def ImageToUint18(self, image):
        """
        将NumPy数组转化为数据是11位整数。
        """
        image = self.normalization(image)

        # 将张量平移并缩放，使其范围在[0, 2047]
        image = (image + 1) * 2047/2.0
        
        # 确保数据在[0, 2047]范围内
        image.clip(0, 2047)
        
        return image.astype(np.uint16)  # 转换为16位整数，以存储11位数据

    def RSGenerate(self,image, percent, colorization=True):
        m, n, c = image.shape
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

    def save_image(self,ms_image, save_dir, bands, flag_cut_bounds, dim_cut, ratio):
        # 保存图像函数，适用于四波段或八波段图像
        #RGB
        if flag_cut_bounds:
            w, h,c = ms_image.shape
            ms_image = ms_image[round(dim_cut/ratio):-round(dim_cut/ratio), 
                                round(dim_cut/ratio):-round(dim_cut/ratio), :]
        selected_bands = ms_image[:, :, bands]
        Rs_img = self.RSGenerate(selected_bands,1,1)
        if flag_cut_bounds:
            new_size = (w,h)
            Rs_img = cv2.resize(Rs_img, new_size)
        imageio.imwrite(save_dir, Rs_img)

    def normalization(self, img):
        # img:  m x n x c
        max_patch = np.max(img, axis=(0, 1), keepdims=True)
        min_patch = np.min(img, axis=(0, 1), keepdims=True)
        img = (img - min_patch) / (max_patch - min_patch)
        return img.astype(np.float32)
    
    def process_image_and_save_tif(self, ms_data, pan_data, output_path,idx):
        """
        处理图像并保存为TIF格式。
        """
        pan = np.transpose(pan_data,(1,2,0))
        ms = np.transpose(ms_data,(1,2,0))
        output_np = self.model(self.normalization(pan),self.normalization(ms))  #unit8 
        output_image = self.ImageToUint18(output_np)
        save_mat = os.path.join(output_path,'mat_img')
        if not os.path.exists(save_mat):
            os.makedirs(save_mat)
        #1 save mat
        output_mat_path = os.path.join(save_mat, f'{idx}.mat')
        mat_dict = {'ms_image': output_image}
        # 保存字典到.mat文件
        savemat(output_mat_path, mat_dict)

        #2 save rgb rs
        save_tif_rs_rgb = os.path.join(output_path,'tif_rs_rgb')
        if not os.path.exists(save_tif_rs_rgb):
            os.makedirs(save_tif_rs_rgb)
        output_rgb_path = os.path.join(save_tif_rs_rgb, f'{idx}.tif')
        _,_,bands  = mat_dict['ms_image'].shape
        if bands == 4: #4 bands
            self.save_image(mat_dict['ms_image'], output_rgb_path, [2, 1, 0], 1, 21, 4)
        else: #8 bands
            self.save_image(mat_dict['ms_image'], output_rgb_path, [4, 2, 1], 1, 21, 4)

    def run(self):
        """
        主运行函数，读取数据，处理图像，保存结果。
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        #1 reduced_examples
        _,ms_data, pan_data = self.read_h5_to_tensors(self.reduced_path)
        for i in range(pan_data.shape[0]):
            output_tif_path = os.path.join(self.save_dir, 'reduced')
            if not os.path.exists(output_tif_path):
                os.makedirs(output_tif_path)
            self.process_image_and_save_tif(ms_data[i], pan_data[i], output_tif_path,i+1)
        print('reduce_examples finished')
        #2 full_examples
        _, ms_data, pan_data = self.read_h5_to_tensors(self.full_path)
        for i in range(pan_data.shape[0]):
            output_tif_path = os.path.join(self.save_dir, 'full')
            if not os.path.exists(output_tif_path):
                os.makedirs(output_tif_path)
            self.process_image_and_save_tif(ms_data[i], pan_data[i], output_tif_path,i+1)
        print('full_examples finished')

# model_dict= {
#             'EXP':EXP,'Brovey':Brovey,'PCA':PCA,'IHS':IHS,'SFIM':SFIM,
#             'GS':GS,'Wavelet':Wavelet,'MTF_GLP':MTF_GLP,'MTF_GLP_HPM':MTF_GLP_HPM,
#             'GSA':GSA,'CNMF':CNMF,'GFPCA':GFPCA}

Method='GFPCA'
datasets = ['gf2','qb','wv2','wv3']
for name in datasets:
    print(name,':')
    image_test = ImageTest(Method,name)
    image_test.run()


