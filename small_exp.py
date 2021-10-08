import nibabel as nib
import numpy as np
import torch.nn as nn

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
filePET = nib.load("./dataset/petTr/RS_011.nii.gz")
dataPET = filePET.get_fdata()
normPET = normX(dataPET)
halfPET = normPET[:, :, :700]
print(halfPET.shape)
# 700x700x700 343 patches

# try to input it to the UNETR model
# H*W*D*C -> patch (N*N*N)
# L = H*W*D*C / N^3
model = UNETR(
    in_channels=1,
    out_channels=1,
    img_size=(100, 100, 100),
    feature_size=16,
    hidden_size=768,
    mlp_dim=343,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
)
model.add_module("linear", nn.Linear(in_features = opt.block_size, 
                                     out_features = opt.block_size)).to(device)

# try to get the output
