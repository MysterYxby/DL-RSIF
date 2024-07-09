'''
@write the index to excel (reduced)
@author： xu
@time:2024/6.30
'''

import numpy as np
import os
import pandas as pd
from scipy.io import loadmat
from cv2 import PSNR as calculatePSNR
import cv2

def RMSE(I_ms,I_f):
    I_ms = I_ms/2047
    I_f = I_f/2047
    f, ms = I_f.astype(np.float32), I_ms.astype(np.float32)
    h, w, c = f.shape
    D = np.power(ms - f,2)
    rmse = np.sqrt(np.sum(D)/h/w/c)
    return rmse


def RASE(I_ms,I_f):
    I_ms = I_ms/2047
    I_f = I_f/2047
    f, ms = I_f.astype(np.float32), I_ms.astype(np.float32)
    h, w, c = f.shape
    C1 = np.sum(np.power(ms[:, :, 0] - f[:, :, 0], 2)) / h / w
    C2 = np.sum(np.power(ms[:, :, 1] - f[:, :, 1], 2)) / h / w
    C3 = np.sum(np.power(ms[:, :, 2] - f[:, :, 2], 2)) / h / w
    C4 = np.sum(np.power(ms[:, :, 3] - f[:, :, 3], 2)) / h / w
    rase = np.sqrt((C1+C2+C3+C4)/4) * 100 / np.mean(ms)
    return rase


def QAVE(I_ms,I_f):
    I_ms = I_ms/2047
    I_f = I_f/2047
    f, ms = I_f.astype(np.float32), I_ms.astype(np.float32)
    h, w, c = f.shape
    ms_mean = np.mean(ms,axis=-1)
    f_mean = np.mean(f,axis=-1)
    M1 = ms[:,:,0] - ms_mean
    M2 = ms[:,:,1] - ms_mean
    M3 = ms[:,:,2] - ms_mean
    M4 = ms[:,:,3] - ms_mean
    F1 = f[:, :, 0] - f_mean
    F2 = f[:, :, 1] - f_mean
    F3 = f[:, :, 2] - f_mean
    F4 = f[:, :, 3] - f_mean
    Qx = (1/c - 1) * (np.power(M1,2) + np.power(M2,2) + np.power(M3,2) + np.power(M4,2))
    Qy = (1/c - 1) * (np.power(F1,2) + np.power(F2,2) + np.power(F3,2) + np.power(F4,2))
    Qxy = (1/c - 1) * (M1 * F1 + M2 * F2 + M3 * F3 + M4 * F4)
    Q = (c * Qxy * ms_mean * f_mean) / ( (Qx + Qy) * ( np.power(ms_mean,2) + np.power(f_mean,2) ) + 2.2204e-16)
    qave = np.sum(Q) / h / w
    return qave


def _ssim(img1, img2, dynamic_range=2047):
    """SSIM for 2D (one-band) image, shape (H, W); uint8 if 225; uint16 if 2047"""
    C1 = (0.01 * dynamic_range)**2
    C2 = (0.03 * dynamic_range)**2
    
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)  # kernel size 11
    window = np.outer(kernel, kernel.transpose())
    
    mu1 = cv2.filter2D(img1_, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2_, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1_**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2_**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1_ * img2_, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def ssim(img1, img2, dynamic_range=2047):
    """SSIM for 2D (H, W) or 3D (H, W, C) image; uint8 if 225; uint16 if 2047"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return _ssim(img1, img2, dynamic_range)
    elif img1.ndim == 3:
        ssims = [_ssim(img1[..., i], img2[..., i], dynamic_range) for i in range(img1.shape[2])]
        return np.array(ssims).mean()
    else:
        raise ValueError('Wrong input image dimensions.')

def ERGAS(img_fake, img_real, scale=4):
    """ERGAS for 2D (H, W) or 3D (H, W, C) image; uint or float [0, 1].
    scale = spatial resolution of PAN / spatial resolution of MUL, default 4."""
    if not img_fake.shape == img_real.shape:
        raise ValueError('Input images must have the same dimensions.')
    img_fake_ = img_fake.astype(np.float64)
    img_real_ = img_real.astype(np.float64)
    if img_fake_.ndim == 2:
        mean_real = img_real_.mean()
        mse = np.mean((img_fake_ - img_real_)**2)
        return 100 / scale * np.sqrt(mse / (mean_real**2 + np.finfo(np.float64).eps))
    elif img_fake_.ndim == 3:
        means_real = img_real_.reshape(-1, img_real_.shape[2]).mean(axis=0)
        mses = ((img_fake_ - img_real_)**2).reshape(-1, img_fake_.shape[2]).mean(axis=0)
        return 100 / scale * np.sqrt((mses / (means_real**2 + np.finfo(np.float64).eps)).mean())
    else:
        raise ValueError('Wrong input image dimensions.')

def SAM(img1, img2):
    """SAM for 3D image, shape (H, W, C); uint or float[0, 1]"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    assert img1.ndim == 3 and img1.shape[2] > 1, "image n_channels should be greater than 1"
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    inner_product = (img1_ * img2_).sum(axis=2)
    img1_spectral_norm = np.sqrt((img1_**2).sum(axis=2))
    img2_spectral_norm = np.sqrt((img2_**2).sum(axis=2))
    # numerical stability
    cos_theta = (inner_product / (img1_spectral_norm * img2_spectral_norm + np.finfo(np.float64).eps)).clip(min=0, max=1)
    return np.mean(np.arccos(cos_theta))


def SCC(img1, img2):
    """SCC for 2D (H, W)or 3D (H, W, C) image; uint or float[0, 1]"""
    if not  img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    if img1_.ndim == 2:
        return np.corrcoef(img1_.reshape(1, -1), img2_.rehshape(1, -1))[0, 1]
    elif img1_.ndim == 3:
        #print(img1_[..., i].reshape[1, -1].shape)
        #test = np.corrcoef(img1_[..., i].reshape[1, -1], img2_[..., i].rehshape(1, -1))
        #print(type(test))
        ccs = [np.corrcoef(img1_[..., i].reshape(1, -1), img2_[..., i].reshape(1, -1))[0, 1]
               for i in range(img1_.shape[2])]
        return np.mean(ccs)
    else:
        raise ValueError('Wrong input image dimensions.')

def _qindex(img1, img2, block_size=8):
    """Q-index for 2D (one-band) image, shape (H, W); uint or float [0, 1]"""
    assert block_size > 1, 'block_size shold be greater than 1!'
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    window = np.ones((block_size, block_size)) / (block_size**2)
    # window_size = block_size**2
    # filter, valid
    pad_topleft = int(np.floor(block_size/2))
    pad_bottomright = block_size - 1 - pad_topleft
    mu1 = cv2.filter2D(img1_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu2 = cv2.filter2D(img2_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1_**2, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu1_sq
    sigma2_sq = cv2.filter2D(img2_**2, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu2_sq
    #print(mu1_mu2.shape)
    #print(sigma2_sq.shape)
    sigma12 = cv2.filter2D(img1_ * img2_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu1_mu2

    # all = 1, include the case of simga == mu == 0
    qindex_map = np.ones(sigma12.shape)
    # sigma == 0 and mu != 0
    idx = ((sigma1_sq + sigma2_sq) == 0) * ((mu1_sq + mu2_sq) != 0)
    qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
    # sigma !=0 and mu == 0
    idx = ((sigma1_sq + sigma2_sq) != 0) * ((mu1_sq + mu2_sq) == 0)
    qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
    # sigma != 0 and mu != 0
    idx = ((sigma1_sq + sigma2_sq) != 0) * ((mu1_sq + mu2_sq) != 0)
    qindex_map[idx] =((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
        (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])
    return np.mean(qindex_map)

def qindex(img1, img2, block_size=8):
    """Q-index for 2D (H, W) or 3D (H, W, C) image; uint or float [0, 1]"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return _qindex(img1, img2, block_size)
    elif img1.ndim == 3:
        qindexs = [_qindex(img1[..., i], img2[..., i], block_size) for i in range(img1.shape[2])]
        return np.array(qindexs).mean()
    else:
        raise ValueError('Wrong input image dimensions.')


# 计算PSNR值
def calculatePSNR(A, B, L):
    peakval = 2**L
    mse = np.mean((A.astype(np.float32) - B.astype(np.float32))**2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(peakval**2 / mse)

def corr2(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    r = np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))
    return r


# 评价指标函数
def coumpute_index(Method_name, method_result_name, dataset_name, dtype_name):
    ratio = 4  # Resize Factor
    L = 11  # Radiometric Resolution
    base_fusion_dir = 'output'
    base_gt_dir = 'image'
    for i in range(len(method_result_name)):
        fusion_dir = os.path.join(base_fusion_dir, Method_name, method_result_name[i], dtype_name, 'mat_img')
        gt_dir = os.path.join(base_gt_dir, dataset_name[i], dtype_name, 'gt')
        path = f'index/{Method_name}'
        excel_filename = f'index/{Method_name}/{dataset_name[i]}_{dtype_name}.xlsx'
        if not os.path.exists(path):
            os.makedirs(path)
        # 检查文件是否存在
        if not os.path.exists(excel_filename):
            # 文件不存在，创建一个新的空Excel文件
            empty_df = pd.DataFrame()
            empty_df.to_excel(excel_filename, index=False, header=False)

        # 初始化评价指标列表
        all_q_values = []
        all_sam_values = []
        all_ergas_values = []
        all_scc_values = []
        all_ssim_values = []
        all_psnr_values = []
        all_rmse_values = []
        all_rase_values = []
        all_cc_values = []

        # 获取文件夹下的所有.mat文件
        files_fusion = [f for f in os.listdir(fusion_dir) if f.endswith('.mat')]
        files_gt = [f for f in os.listdir(gt_dir) if f.endswith('.mat')]

        # 确保files_gt和files_fusion的顺序是对应的
        assert len(files_fusion) == len(files_gt), "The number of fusion and GT files must match."

        # 写入标题行
        column_names = ['Num', 'Q', 'SAM', 'ERGAS', 'SCC', 'SSIM', 'PSNR','RMES','RASE','CC']

        # 创建一个新的DataFrame
        df = pd.DataFrame(columns=column_names)

        # 写入'Ideal'行
        ideal_row = {'Num': 'Ideal', 'Q': 1, 'SAM': 0, 'ERGAS': 0, 'SCC': 1, 'SSIM': 1, 'PSNR': '+','RMES':0,'RASE':0,'CC':1}
        df = df.append(ideal_row, ignore_index=True)

        # 循环处理每个文件
        for m, (file_fusion, file_gt_name) in enumerate(zip(files_fusion, files_gt)):
            file_path_fusion = os.path.join(fusion_dir, file_fusion)
            file_path_gt = os.path.join(gt_dir, file_gt_name)

            # 加载.mat文件
            fusion = loadmat(file_path_fusion)
            gt = loadmat(file_path_gt)

            fusion_img = fusion['ms_image']
            gt_img = gt['gt']

            # 计算评价指标
            Q_index = qindex(gt_img, fusion_img)
            SAM_index = SAM(gt_img, fusion_img)
            ERGAS_index = ERGAS(gt_img, fusion_img, ratio)
            SCC_index = SCC(fusion_img, gt_img)
            SSIM_index = ssim(fusion_img, gt_img)
            PSNR_index = calculatePSNR(gt_img, fusion_img, L)
            RMSE_index = RMSE(gt_img, fusion_img)
            RASE_index = RASE(gt_img, fusion_img)
            CC_index = corr2(gt_img, fusion_img)
            # 收集评价指标
            all_q_values.append(Q_index)
            all_sam_values.append(SAM_index)
            all_ergas_values.append(ERGAS_index)
            all_scc_values.append(SCC_index)
            all_ssim_values.append(SSIM_index)
            all_psnr_values.append(PSNR_index)
            all_rmse_values.append(RMSE_index)
            all_rase_values.append(RASE_index)
            all_cc_values.append(CC_index)
            # 写入每张图像的评价指标为一行
            row = {
                'Num': m + 1,
                'Q': Q_index,
                'SAM': SAM_index,
                'ERGAS': ERGAS_index,
                'SCC': SCC_index,
                'SSIM': SSIM_index,
                'PSNR': PSNR_index,
                'RMES':RMSE_index,
                'RASE':RASE_index,
                'CC':CC_index
            }
            df = df.append(row, ignore_index=True)

        # 计算并写入平均值和标准差
        mean_row = {
            'Num': 'Mean',
            'Q': np.mean(all_q_values),
            'SAM': np.mean(all_sam_values),
            'ERGAS': np.mean(all_ergas_values),
            'SCC': np.mean(all_scc_values),
            'SSIM': np.mean(all_ssim_values),
            'PSNR': np.mean(all_psnr_values),
            'RMES':np.mean(all_rmse_values),
            'RASE':np.mean(all_rase_values),
            'CC':np.mean(all_cc_values)
        }
        df = df.append(mean_row, ignore_index=True)

        std_row = {
            'Num': 'Std',
            'Q': np.std(all_q_values),
            'SAM': np.std(all_sam_values),
            'ERGAS': np.std(all_ergas_values),
            'SCC': np.std(all_scc_values),
            'SSIM': np.std(all_ssim_values),
            'PSNR': np.std(all_psnr_values),
            'RMES':np.std(all_rmse_values),
            'RASE':np.std(all_rase_values),
            'CC':np.std(all_cc_values)
        }
        df = df.append(std_row, ignore_index=True)

        # 保存到Excel文件
        df.to_excel(excel_filename, index=False)
        print(f'{dataset_name[i]} finished!')


if __name__ == '__main__':
    # 假设这里是你的主函数或脚本的开始
    Method_name = 'BDPN'
    method_result_name = ['gf2', 'qb', 'wv2', 'wv3']
    dataset_name = ['Gaofen2', 'QuickBird', 'WorldView2', 'WorldView3']
    dtype_name = 'reduced'
    coumpute_index(Method_name, method_result_name, dataset_name, dtype_name)