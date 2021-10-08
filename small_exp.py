import nibabel as nib
import numpy as np
import torch.nn as nn

import torch
import glob
import os

from monai.networks.nets import UNETR
from torch.nn import Linear

def normX(data):
    data[data<0] = 0
    data[data>3000] = 6000  
    data = data / 6000
    return data

# try to load one data
filePET = nib.load("./dataset/sctTr/CUB_011.nii.gz")
dataPET = filePET.get_fdata()[:, :, -512:]
normPET = normX(dataPET)
normPET = np.expand_dims(normPET, axis=(0, 1))
print(normPET.shape)
# 700x700x700 343 patches
# 3dresample -dxyz 1.367 1.367 1.367 -prefix CUB_011.nii.gz -input CT_011.nii.gz

# try to input it to the UNETR model
# H*W*D*C -> patch (N*N*N)
# L = H*W*D*C / N^3
device = torch.device("cuda")
model = UNETR(
    in_channels=1,
    out_channels=1,
    img_size=(512, 512, 512),
    feature_size=64,
    hidden_size=1024,
    mlp_dim=4096,
    num_heads=16,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).float().to(device)

inputPET = torch.from_numpy(normPET).float().to(device)
outputPET = model(inputPET)
print(outputPET.size())
# model.add_module("linear", nn.Linear(in_features = opt.block_size, 
#                                      out_features = opt.block_size)).to(device)

# try to get the output
