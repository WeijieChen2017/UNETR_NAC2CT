# 3dresample -dxyz 1.367 1.367 1.367 -prefix CUB_011.nii.gz -input CT_011.nii.gz
import nibabel as nib
import numpy as np
import torch.nn as nn

import torch
import glob
import time
import os

from monai.networks.nets import UNETR
from torch.nn import Linear

def normX(data):
    data[data<0] = 0
    data[data>3000] = 6000  
    data = data / 6000
    return data

batch_size = 16


# device = torch.device("cuda")
# model = UNETR(
#     in_channels=1,
#     out_channels=1,
#     img_size=(lx, ly, lz),
#     feature_size=256,
#     hidden_size=4096,
#     mlp_dim=4096,
#     num_heads=16,
#     pos_embed="perceptron",
#     norm_name="instance",
#     res_block=True,
#     dropout_rate=0.0,
# ).half().to(device)

# loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
# torch.backends.cudnn.benchmark = True
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
# model.train()


# try to load one data
filePET = nib.load("./dataset/sctTr/CUB_011.nii.gz")
# dataPET = filePET.get_fdata()[:, :, idx-8:idx+8]
dataPET = filePET.get_fdata()[:, :, -512:]
normPET = normX(dataPET)
normPET = np.expand_dims(normPET, axis=(0, 1))
hx, hy, hz = normPET.shape
lx, ly, lz = hx//2, hy//2, hz//2
normPET = np.resize(normPET, (lx, ly, lz))
print(normPET.shape)

cntBatch = 0
inputBatch = np.zeros((batch_size, 1, hx, hy, 8))

for idz in range(hz//8):
	sliceBatch = normPET[:, :, :, :, idz*8 : idz*8+8]
	print(sliceBatch.shape)
	inputBatch[cntBatch, :, :, :, :] = sliceBatch
	cntBatch += 1

	if cntBatch == 16:
		print(inputBatch.shape)
		# train an epoch
		cntBatch = 0



# try to input it to the UNETR model
# H*W*D*C -> patch (N*N*N)
# L = H*W*D*C / N^3


# inputPET = torch.from_numpy(normPET).half().to(device)
# outputPET = model(inputPET)
# print(outputPET.size())
# model.add_module("linear", nn.Linear(in_features = opt.block_size, 
#                                      out_features = opt.block_size)).to(device)

# try to get the output
