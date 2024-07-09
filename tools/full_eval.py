'''
@write the index to excel (full)
@author： xu
@time:2024/6.30
'''
import numpy as np
import scipy.signal
import scipy.io
import cv2
import os
from scipy.io import loadmat
import pandas as pd

class eval_full_QualityIndex:
    def __init__(self,sensor,method):
        self.sensor = sensor
        self.method = method
        self.filter = self.GetMTF_Filter()
        self.dataset = {
            'qb': 'QuickBird',
            'gf2': 'Gaofen2',
            'wv2': 'WorldView2',
            'wv3': 'WorldView3'
        }
        self.fusion_dir = os.path.join('output', self.method,self.sensor,'full','mat_img')
        
        #origin dataset
        self.MS_dir = os.path.join('image', self.dataset[self.sensor],'full','MS')
        self.PAN_dir = os.path.join('image', self.dataset[self.sensor],'full','PAN')
        self.save_dir = os.path.join('index', self.method)

    def create_excel(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        excel_name = self.dataset[self.sensor] + '_full.xlsx'
        # 使用Pandas创建一个空的DataFrame
        columns = ['Num', 'D_lambda', 'D_s', 'QNR']
        df = pd.DataFrame(columns=columns)
        excel_path = os.path.join(self.save_dir, excel_name)
        
        # 写入标题和理想情况行
        df.loc[0] = ['Ideal', 0, 0, 1]
        df.to_excel(excel_path, index=False)
        print(f"Excel文件已创建：{excel_path}")
        return excel_path, df
    

    def compute_index(self):
        #MS PAN FUSION mat文件名一样
        # excel = self.create_excel()
        excel_path,df = self.create_excel()
        # 存储所有.mat文件的路径
        fusion_mat_files = [os.path.join(self.fusion_dir, f) for f in os.listdir(self.fusion_dir) if f.endswith('.mat')]
        ms_mat_files = [os.path.join(self.MS_dir, f) for f in os.listdir(self.MS_dir) if f.endswith('.mat')]
        pan_mat_files = [os.path.join(self.PAN_dir, f) for f in os.listdir(self.PAN_dir) if f.endswith('.mat')]
        D_list1 = []
        D_list2 = []
        QNR_list = []
        # 遍历所有.mat文件
        for i, (fusion_mat, ms_mat, pan_mat) in enumerate(zip(fusion_mat_files, ms_mat_files, pan_mat_files)):
            # ... 加载.mat文件的代码保持不变 ...
            fusion_data = loadmat(fusion_mat_files[i])
            ms_data = loadmat(ms_mat_files[i])
            pan_data = loadmat(pan_mat_files[i])
            F = fusion_data['ms_image']
            MS = ms_data['ms']
            PAN = pan_data['pan']


            # 计算指标
            D_lambda_index, D_s_index, QNR_index = self.QNR(F, MS, PAN)
            info = {'Num': i+1, 'D_lambda': D_lambda_index, 'D_s': D_s_index, 'QNR': QNR_index}
            df = df.append(info, ignore_index=True)
            
            # 收集数据以计算最终的统计值
            D_list1.append(D_lambda_index)
            D_list2.append(D_s_index)
            QNR_list.append(QNR_index)

        # 计算平均值和标准差
        df.loc[len(df)] = ['Mean', np.mean(D_list1), np.mean(D_list2), np.mean(QNR_list)]
        df.loc[len(df)] = ['Std', np.std(D_list1), np.std(D_list2), np.std(QNR_list)]
        # 保存到Excel文件
        df.to_excel(excel_path, index=False)
        print(f'{self.dataset[self.sensor]} finished')


    def GetMTF_Filter(self):
        if self.sensor == 'qb':
            MTF_Filter = scipy.io.loadmat('./MTF_PAN/QBfilter.mat')['QBfilter']
        elif self.sensor == 'IKONOS':
            MTF_Filter = scipy.io.loadmat('./MTF_PAN/IKONOSfilter.mat')['IKONOSfilter']
        elif self.sensor == 'GeoEye1':
            MTF_Filter = scipy.io.loadmat('./MTF_PAN/GeoEye1filter.mat')['GeoEye1filter']
        elif self.sensor == 'wv2':
            MTF_Filter = scipy.io.loadmat('./MTF_PAN/WV2filter.mat')['WV2filter']
        else:
            MTF_Filter = scipy.io.loadmat('./MTF_PAN/nonefilter.mat')['nonefilter']
            pass
        return MTF_Filter

    def MTF_PAN(self,image_pan):
        pan = np.pad(image_pan,((20,20),(20,20)),mode='edge')
        image_pan_filter = scipy.signal.correlate2d(pan,self.filter,mode='valid')
        pan_filter = (image_pan_filter + 0.5).astype(np.uint8).astype(np.float32)
        return pan_filter

    def UQI(self,x,y):
        x = x.flatten()
        y = y.flatten()
        mx = np.mean(x)
        my = np.mean(y)
        C = np.cov(x, y)
        Q = 4 * C[0, 1] * mx * my / (C[0,0] + C[1, 1] + 1e-21) / (mx**2 + my**2 + 1e-21)
        return Q

    def D_s(self,fusion,ms,pan,S,q):
        D_s_index = 0
        h, w, c = fusion.shape
        pan_filt = self.MTF_PAN(pan)

        for i in range(c):
            band_fusion = fusion[:,:,i]
            band_pan = pan
            # 分块
            Qmap_high = []
            for y in range(0,h,S):
                for x in range(0,w,S):
                    Qmap_high.append(self.UQI(band_fusion[y:y+S,x:x+S],band_pan[y:y+S,x:x+S]))
                    pass
                pass
            Q_high = np.mean(np.asarray(Qmap_high))

            band_ms = ms[:, :, i]
            band_pan_filt = pan_filt
            # 分块
            Qmap_low = []
            for y in range(0, h, S):
                for x in range(0, w, S):
                    Qmap_low.append(self.UQI(band_ms[y:y + S, x:x + S], band_pan_filt[y:y+S,x:x+S]))
                    pass
                pass
            Q_low = np.mean(np.asarray(Qmap_low))
            D_s_index = D_s_index + np.abs(Q_high - Q_low)**q

        D_s_index = (D_s_index / c)**(1 / q)
        return D_s_index

    def D_lambda(self,fusion,ms,S,p):
        D_lambda_index = 0
        h, w, c = fusion.shape
        for i in range(0,c-1):
            for j in range(i+1,c):
                bandi = ms[:,:,i]
                bandj = ms[:,:,j]
                # 分块
                Qmap_exp = []
                for y in range(0, h, S):
                    for x in range(0, w, S):
                        Qmap_exp.append(self.UQI(bandi[y:y + S, x:x + S], bandj[y:y + S, x:x + S]))
                        pass
                    pass
                Q_exp = np.mean(np.asarray(Qmap_exp))

                bandi = fusion[:, :, i]
                bandj = fusion[:, :, j]
                # 分块
                Qmap_fused = []
                for y in range(0, h, S):
                    for x in range(0, w, S):
                        Qmap_fused.append(self.UQI(bandi[y:y + S, x:x + S], bandj[y:y + S, x:x + S]))
                        pass
                    pass
                Q_fused = np.mean(np.asarray(Qmap_fused))
                D_lambda_index = D_lambda_index + np.abs(Q_fused - Q_exp)**p

        s = (c**2 - c)/2
        D_lambda_index = (D_lambda_index/s)**(1/p)
        return D_lambda_index

    def QNR(self,fusion,ms,pan,S = 32,p = 1,q = 1,alpha = 1,beta = 1):
        # The size of the fusion, pan and ms is the same
        # ms (Use bicubic methoc to upsample)
        # The difference between the matlab and python comes from interpolation method
        h, w, c = fusion.shape
        ms_upsample = cv2.resize(ms,dsize=(w,h),interpolation=cv2.INTER_CUBIC)
        D_lambda_index = self.D_lambda(fusion,ms_upsample,S,p)
        D_s_index = self.D_s(fusion,ms_upsample,pan,S,q)
        QNR_index = ((1 - D_lambda_index)**alpha) * ((1 - D_s_index)**beta)
        return D_lambda_index,D_s_index,QNR_index


if __name__ == '__main__':
    method = 'PNN'
    sensors = ['qb','gf2','wv2','wv3']
    for sensor in sensors:
        Eval_method = eval_full_QualityIndex(sensor,method)
        Eval_method.compute_index()
