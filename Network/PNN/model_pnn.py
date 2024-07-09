import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class PNN(nn.Module):
    def __init__(self, spectral_num, channel=64):
        super(PNN, self).__init__()

        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.conv1 = nn.Conv2d(in_channels=spectral_num + 1, out_channels=channel, kernel_size=9, stride=1,padding = 4,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=32, kernel_size=5, stride=1,padding = 2,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=spectral_num, kernel_size=5, stride=1,padding = 2,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

        # init_weights(self.conv1, self.conv2, self.conv3)

    def forward(self, x):  # x = cat(lms,pan)
        input1 = x  # Bsx9x64x64

        rs = self.relu(self.conv1(input1))
        rs = self.relu(self.conv2(rs))
        output = self.conv3(rs)

        return output

    @classmethod
    def set_blk(cls, blk):
        cls.blk = blk

# ----------------- End-Main-Part ------------------------------------
