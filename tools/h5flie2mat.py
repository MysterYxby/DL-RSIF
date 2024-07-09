'''
@this is py for convert the reduced and full examples files to mat to compute the index
@ autor: xu
@ time: 2024.6.28
'''
import torch
import numpy as np
import h5py
import os 
from scipy.io import savemat

def read_h5_to_tensors(h5_file_path):
        """
        读取HDF5文件中的多个数据集，并转换为tensors。
        """
        '''
        lms: 上采用多光谱图像   256x256
        ms:  多光谱图像         64x64
        pan: pan图像           256x256
        gt: 未下采样MS图像      256x256
        '''
        with h5py.File(h5_file_path, 'r') as h5_file:
            keys = list(h5_file.keys())
            # print('gt' in keys)
            # print(keys)
            lms_data = torch.from_numpy(h5_file['lms'][:])
            ms_data = torch.from_numpy(h5_file['ms'][:])
            pan_data = torch.from_numpy(h5_file['pan'][:])
            # lms_data = h5_file['lms'][:]
            # ms_data = h5_file['ms'][:]
            # pan_data = h5_file['pan'][:]
            if 'gt' not in keys:
                return lms_data, ms_data, pan_data 
            else:
                 gt_data = torch.from_numpy(h5_file['gt'][:])
                #  gt_data = h5_file['gt'][:]
                 return lms_data, ms_data, pan_data, gt_data
            
        
def TensorToImage(image_tensor):
    """
    将PyTorch tensor转换为图像格式的NumPy数组，并确保数据是11位整数。
    """
    # 确保输入是一个PyTorch张量
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("image_tensor must be a PyTorch tensor")
    
    # 获取张量的维度
    c, m, n = image_tensor.size(-3), image_tensor.size(-2), image_tensor.size(-1)
    
    # 将张量平移并缩放，使其范围在[0, 2047]
    image_tensor = (image_tensor + 1) * 2047/2.0
    
    # 确保数据在[0, 2047]范围内
    image_tensor.clamp_(0, 2047)
    
    # 将张量转换为NumPy数组
    if c == 1:
        image_np = image_tensor.detach().numpy().reshape(m, n)
    else:
        image_np = np.zeros((m, n, c), dtype=np.uint16)
        for i in range(c):
            image_np[:, :, i] = image_tensor[i].detach().numpy()
    
    return image_np.astype(np.uint16)  # 转换为16位整数，以存储11位数据

def h5tomat(tensor,save_dir,type_img):
    tensor_np = TensorToImage(tensor)
    mat_dict = {type_img: tensor_np}
    savemat(save_dir, mat_dict)
    

if __name__ =='__main__':
    full = {False, True}
    for i in full:
        if i:
            h5_dir = 'D:/Pan/WorldView3/full_examples/test_wv3_OrigScale_multiExm1.h5'
            PAN_save = 'image/WorldView3/full/PAN'
            GT_save = 'image/WorldView3/full/GT'
            MS_save = 'image/WorldView3/full/MS'
            LMS_save = 'image/WorldView3/full/LMS'
    
            lms_data, ms_data, pan_data =read_h5_to_tensors(h5_dir)
            for i in range(lms_data.shape[0]):
                PAN_save_dir = os.path.join(PAN_save,f'{i+1}.mat')
                GT_save_dir = os.path.join(GT_save,f'{i+1}.mat')
                MS_save_dir = os.path.join(MS_save,f'{i+1}.mat')
                LMS_save_dir = os.path.join(LMS_save,f'{i+1}.mat')
                h5tomat(pan_data[i].div(2047),PAN_save_dir,'pan')
                h5tomat(ms_data[i].div(2047),MS_save_dir,'ms')
                h5tomat(lms_data[i].div(2047),LMS_save_dir,'lms')
        else:
            h5_dir = 'D:/Pan/WorldView3/reduced_examples/test_wv3_multiExm1.h5'
            PAN_save = 'image/WorldView3/reduced/PAN'
            GT_save = 'image/WorldView3/reduced/GT'
            MS_save = 'image/WorldView3/reduced/MS'
            LMS_save = 'image/WorldView3/reduced/LMS'
            lms_data, ms_data, pan_data, gt_data =read_h5_to_tensors(h5_dir)
            for i in range(lms_data.shape[0]):
                PAN_save_dir = os.path.join(PAN_save,f'{i+1}.mat')
                GT_save_dir = os.path.join(GT_save,f'{i+1}.mat')
                MS_save_dir = os.path.join(MS_save,f'{i+1}.mat')
                LMS_save_dir = os.path.join(LMS_save,f'{i+1}.mat')
                h5tomat(pan_data[i].div(2047),PAN_save_dir,'pan')
                h5tomat(gt_data[i].div(2047),GT_save_dir,'gt')
                h5tomat(ms_data[i].div(2047),MS_save_dir,'ms')
                h5tomat(lms_data[i].div(2047),LMS_save_dir,'lms')
    print('done!')
          