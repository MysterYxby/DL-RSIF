import torch
import torch.nn as nn
import imageio
import numpy as np
import h5py
import os
import cv2
from model_pannet import PanNet  
from scipy.io import savemat

class ImageTest:
    def __init__(self, device,dataset_name):
        self.dataset_name = dataset_name
        self.bands = {
            'qb': 4,
            'gf2': 4,
            'wv2': 8,
            'wv3': 8
        }
        self.scale = 2047
        self.device = device
        self.model = PanNet(spectral_num = self.bands[self.dataset_name]).to(self.device)
        self.path = {
            #qb
            'qb_model_path':'pretrained-model/QB/pannet.pth',    #4bands
            'qb_reduced_dir': 'D:/Pan/QuickBird/reduced_examples/test_qb_multiExm1.h5',
            'qb_full_dir': 'D:/Pan/QuickBird/full_examples/test_qb_OrigScale_multiExm1.h5',
            'qb_fusion_dir':'output/PanNet/qb',
            #gf
            'gf2_model_path':'pretrained-model/QB/pannet.pth',    #4bands
            'gf2_reduced_dir': 'D:/Pan/Gaofen2/reduced_examples/test_gf2_multiExm1.h5',
            'gf2_full_dir': 'D:/Pan/Gaofen2/full_examples/test_gf2_OrigScale_multiExm1.h5',
            'gf2_fusion_dir':'output/PanNet/gf2',
            #wv2
            'wv2_model_path':'pretrained-model/WV2/pannet.pth',    #8bands
            'wv2_reduced_dir': 'D:/Pan/WorldView2/reduced_examples/test_wv2_multiExm1.h5',
            'wv2_full_dir': 'D:/Pan/WorldView2/full_examples/test_wv2_OrigScale_multiExm1.h5',
            'wv2_fusion_dir':'output/PanNet/wv2',
            #wv3
            'wv3_model_path':'pretrained-model/WV3/pannet.pth',    #8bands
            'wv3_reduced_dir': 'D:/Pan/WorldView3/reduced_examples/test_wv3_multiExm1.h5',
            'wv3_full_dir': 'D:/Pan/WorldView3/full_examples/test_wv3_OrigScale_multiExm1.h5',
            'wv3_fusion_dir':'output/PanNet/wv3'
        }
        self.model_path, self.reduced_path, self.full_path, self.save_dir = self.load_path()
        self.load_pretrained_model()
    # get high-frequency
    def get_edge(self, data):
        data = data.numpy()
        rs = np.zeros_like(data)
        N = data.shape[0]
        for i in range(N):
            if len(data.shape) == 3:
                rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5),
                                                            normalize=True)  # 第二个参数的-1表示输出图像使用的深度与输入图像相同
            else:
                rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5), normalize=True)
        return torch.from_numpy(rs)

    def load_path(self):
        #dataset_name = qb or gf2 or wv2 or wv3
        model_path_key = self.dataset_name + '_model_path'
        reduced_dir_key = self.dataset_name + '_reduced_dir'
        full_dir_key = self.dataset_name + '_full_dir'
        fusion_dir_key = self.dataset_name + '_fusion_dir'
        return self.path[model_path_key], self.path[reduced_dir_key],self.path[full_dir_key],self.path[fusion_dir_key]
        
    def load_pretrained_model(self):
        """
        加载预训练的模型权重。
        """
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    def read_h5_to_tensors(self, h5_file_path):
        """
        读取HDF5文件中的多个数据集，并转换为PyTorch tensors。
        """
        '''
        lms: 上采用多光谱图像   256x256
        ms:  多光谱图像         64x64
        pan: pan图像           256x256
        gt： 未下采样多光谱图像 256x256
        '''
        with h5py.File(h5_file_path, 'r') as h5_file:
            lms_data = torch.from_numpy(h5_file['lms'][:]).to(self.device)
            ms_data = torch.from_numpy(h5_file['ms'][:]).to(self.device)
            pan_data = torch.from_numpy(h5_file['pan'][:]).to(self.device)
        return lms_data, ms_data, pan_data          
        

    def TensorToImage(self, image_tensor):
        """
        将PyTorch tensor转换为图像格式的NumPy数组，并确保数据是11位整数。
        """
        # 确保输入是一个PyTorch张量
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError("image_tensor must be a PyTorch tensor")
        
        #归一化
        image_tensor = (image_tensor-torch.min(image_tensor)) / (torch.max(image_tensor)-torch.min(image_tensor))

        # 获取张量的维度
        c, m, n = image_tensor.size(-3), image_tensor.size(-2), image_tensor.size(-1)
        
        # 将张量平移并缩放，使其范围在[0, 2047]
        image_tensor = (image_tensor + 1) * self.scale/2.0
        
        # 确保数据在[0, 2047]范围内
        image_tensor.clamp_(0, self.scale)
        
        # 将张量转换为NumPy数组
        if c == 1:
            image_np = image_tensor.detach().numpy().reshape(m, n)
        else:
            image_np = np.zeros((m, n, c), dtype=np.uint16)
            for i in range(c):
                image_np[:, :, i] = image_tensor[i].detach().numpy()
        
        return image_np.astype(np.uint16)  # 转换为16位整数，以存储11位数据
    
    def RSGenerate(self,image, percent, colorization=True):
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

    def process_image_and_save_tif(self,lms_data, ms_data, pan_data, output_path,idx):
        """
        处理图像并保存为TIF格式。
        """
        ms_tensor = ms_data.float().div(self.scale)
        pan_tensor = pan_data.float().div(self.scale)
        lms_tensor = lms_data.float()
        with torch.no_grad():
            output_tensor = self.model(self.get_edge(ms_tensor.unsqueeze(0)),self.get_edge(pan_tensor.unsqueeze(0))) + lms_tensor.unsqueeze(0)
        output_image = self.TensorToImage(output_tensor.squeeze(0))
        save_mat = os.path.join(output_path,'mat_img')
        if not os.path.exists(save_mat):
            os.makedirs(save_mat)
        #1 save mat
        output_mat_path = os.path.join(save_mat, f'{idx}.mat')
        mat_dict = {'ms_image': output_image}
        # 保存字典到.mat文件
        savemat(output_mat_path, mat_dict)
        # imageio.imwrite(output_mat_path, output_image)

        save_tif_rs_rgb = os.path.join(output_path,'tif_rs_rgb')
        if not os.path.exists(save_tif_rs_rgb):
            os.makedirs(save_tif_rs_rgb)
        output_rgb_path = os.path.join(save_tif_rs_rgb, f'{idx}.tif')
        # print(output_image.shape)
        if self.bands[self.dataset_name] == 4: #4 bands
            output_image_rs = self.RSGenerate(output_image[:, :, [2, 1, 0]], 1, 1)
        else: #8 bands
            output_image_rs = self.RSGenerate(output_image[:, :, [4, 2, 1]], 1, 1)
        #2 save rgb rs
        imageio.imwrite(output_rgb_path, output_image_rs)

    def run(self):
        """
        主运行函数，读取数据，处理图像，保存结果。
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        #1 reduced_examples
        lms_data, ms_data, pan_data = self.read_h5_to_tensors(self.reduced_path)
        for i in range(ms_data.shape[0]):
            output_tif_path = os.path.join(self.save_dir, 'reduced')
            if not os.path.exists(output_tif_path):
                os.makedirs(output_tif_path)
            self.process_image_and_save_tif(lms_data[i],ms_data[i], pan_data[i], output_tif_path,i+1)
        print('reduce_examples finished')
        #2 full_examples
        lms_data, ms_data, pan_data = self.read_h5_to_tensors(self.full_path)
        for i in range(ms_data.shape[0]):
            output_tif_path = os.path.join(self.save_dir, 'full')
            if not os.path.exists(output_tif_path):
                os.makedirs(output_tif_path)
            self.process_image_and_save_tif(lms_data[i],ms_data[i], pan_data[i], output_tif_path,i+1)
        print('full_examples finished')

# 使用示例
if __name__ == '__main__':
    '''
    8 bands: coastal blue, blue, green, yellow, red, red edge, NIR1, and NIR2
    4 bands: blue, green, red, near-infrared (NIR)
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_test = ImageTest(device, dataset_name = 'qb')
    #模型运行
    image_test.run()