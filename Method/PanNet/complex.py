import torch
import torch.nn as nn
import thop
from model_pannet import *

def profile(model, input_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    example1 = torch.rand(1, *input_size[0])  # Create a random input tensor for input 1
    example2 = torch.rand(1, *input_size[1])  # Create a random input tensor for input 2
    flops, params = thop.profile(model.to(device), (example1, example2), verbose=False)
    flops, params = flops / 1_000_000_000.0, params / 1_000_000.0  # Convert to billion and million
    return flops, params

model = PanNet(spectral_num=4)
model.eval()


# 计算 FLOPs
flops, params = profile(model, ((4, 128, 128), (1, 512, 512)))
print(f"Total number of FLOPs: {flops:.3f} G")
print(f"Total number of parameters: {params:.2f} M")