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

def normY(data):
    data[data<-1000] = -1000
    data[data>3000] = 3000
    data = (data + 1000) / 4000
    return data

batch_size = 4
# loss_batch_cnt = 16
widthZ = 16


device = torch.device("cuda")
model = UNETR(
    in_channels=1,
    out_channels=1,
    img_size=(256, 256, widthZ),
    feature_size=64,
    hidden_size=4096,
    mlp_dim=4096,
    num_heads=16,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
)
sizeInput = torch.from_numpy(np.array((256, 256, widthZ)))
sizeOutput = torch.from_numpy(np.array((256, 256, widthZ)))
model.add_module("linear", nn.Linear(in_features = 256, out_features = 256))
model.half().to(device)

criterion = nn.HuberLoss(reduction="mean")
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
model.train()


# try to load one data
filePET = nib.load("./dataset/petTr/CUB_011.nii.gz")
fileSCT = nib.load("./dataset/sctTr/CUB_011.nii.gz")
dataPET = filePET.get_fdata()[:, :, -512:]
dataSCT = fileSCT.get_fdata()[:, :, -512:]
normPET = normX(dataPET)
normSCT = normY(dataSCT)
hx, hy, hz = normPET.shape
lx, ly, lz = hx//2, hy//2, hz//2
normPET = np.resize(normPET, (lx, ly, lz))
normSCT = np.resize(normSCT, (lx, ly, lz))
normPET = np.expand_dims(normPET, axis=(0, 1))
normSCT = np.expand_dims(normSCT, axis=(0, 1))
print(normPET.shape, normSCT.shape)

cntBatch = 0
inputBatchX = np.zeros((batch_size, 1, lx, ly, widthZ))
inputBatchY = np.zeros((batch_size, 1, lx, ly, widthZ))
print(inputBatchX.shape, inputBatchY.shape)
lenData = lz // widthZ


# epoch_loss = np.zeros((lenData))
# loss_batch = np.zeros((loss_batch_cnt))

for idz in range(lz//widthZ):
    sliceBatchX = normPET[:, :, :, :, idz*widthZ : (idz+1)*widthZ]
    sliceBatchY = normSCT[:, :, :, :, idz*widthZ : (idz+1)*widthZ]
    # print(sliceBatchX.shape, sliceBatchY.shape)
    inputBatchX[cntBatch, :, :, :, :] = sliceBatchX
    inputBatchY[cntBatch, :, :, :, :] = sliceBatchY
    cntBatch += 1

    if cntBatch == batch_size:
        
        realInputX = torch.from_numpy(inputBatchX).half().to(device)
        realInputY = torch.from_numpy(inputBatchY).half().to(device)
        realOutput = model(realInputX)
        print("==>Output shape: ", realOutput.size())
        optimizer.zero_grad()
        loss = criterion(realOutput, realInputY)
        loss.backward()
        optimizer.step()
        loss_voxel = loss.item()

        loss_mean = np.mean(loss_voxel)
        loss_std = np.std(loss_voxel)
        # print("==>", loss_voxel)
        print("==>==>Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))

        # loss_batch[iteration % loss_batch_cnt] = loss_voxel
        # epoch_loss[iteration] = loss_voxel

        # if iteration % loss_batch_cnt == loss_batch_cnt - 1:
        #     loss_mean = np.mean(loss_batch)
        #     loss_std = np.std(loss_batch)
        #     print("===> Epoch[{}]({}/{}): ".format(epoch + 1, iteration + 1, len(data_loader)), end='')
        #     print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))

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
