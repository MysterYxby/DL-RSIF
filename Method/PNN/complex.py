import torch
import torch.nn as nn
import thop
from model_pnn import *

def profile(model, input_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    example = torch.rand(1, *input_size)  # Create a random input tensor
    flops, params = thop.profile(model.to(device), (example, ), verbose=False)
    flops, params = flops / 1_000_000_000.0, params / 1_000_000.0  # Convert to billion and million
    return flops, params

model = PNN(spectral_num=4)
model.eval()


# 计算 FLOPs
flops, params = profile(model, (5, 128, 128))
print(f"Total number of FLOPs: {flops:.3f} G")
print(f"Total number of parameters: {params:.2f} M")