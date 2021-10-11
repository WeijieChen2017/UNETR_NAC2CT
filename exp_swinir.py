# 3dresample -dxyz 1.367 1.367 1.367 -prefix CUB_011.nii.gz -input CT_011.nii.gz
import nibabel as nib
import numpy as np
import torch.nn as nn

import torch
import glob
import time
import os

from monai.networks.nets import UNETR
from monai.networks.blocks.unetr_block import UnetrBasicBlock
from monai.networks.nets.vit import ViT
from torch.nn import Linear

from SwinIR.models.network_swinir import SwinIR

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
widthZ = 4


device = torch.device("cuda")
model = SwinIR(
        # img_size (int | tuple(int)): Input image size. Default 64
        # patch_size (int | tuple(int)): Patch size. Default: 1
        # in_chans (int): Number of input image channels. Default: 3
        # embed_dim (int): Patch embedding dimension. Default: 96
        # depths (tuple(int)): Depth of each Swin Transformer layer.
        # num_heads (tuple(int)): Number of attention heads in different layers.
        # window_size (int): Window size. Default: 7
        # mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        # qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        # qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        # drop_rate (float): Dropout rate. Default: 0
        # attn_drop_rate (float): Attention dropout rate. Default: 0
        # drop_path_rate (float): Stochastic depth rate. Default: 0.1
        # norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        # ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        # patch_norm (bool): If True, add normalization after patch embedding. Default: True
        # use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        # upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        # img_range: Image range. 1. or 255.
        # upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        # resi_connection: The convolutional block before residual connection. '1conv'/'3conv'

        img_size = (256, 256, widthZ),
        patch_size =  1,
        in_chans = 1,
        embed_dim = 128,
        depths = (6, 6, 6, 6, 6, 6),
        num_heads = (6, 6, 6, 6, 6, 6),
        window_size = 8,
        mlp_ratio = 4,
        qkv_bias = True,
        qk_scale = None,
        drop_rate = 0,
        attn_drop_rate = 0,
        drop_path_rate = 0.1,
        norm_layer = nn.LayerNorm,
        ape = True,
        patch_norm = True,
        use_checkpoint = False,
        upscale = 2,
        img_range = 1,
        upsampler = nearest+conv,
        resi_connection = 1conv
        )
sizeInput = torch.from_numpy(np.array((256, 256, widthZ)))
sizeOutput = torch.from_numpy(np.array((256, 256, widthZ)))
model.add_module("linear", nn.Linear(in_features = 256, out_features = 256))

model.vit.register_full_backward_hook(hook_backward_fn)
model.encoder1.register_full_backward_hook(hook_backward_fn)
model.encoder2.register_full_backward_hook(hook_backward_fn)
model.encoder3.register_full_backward_hook(hook_backward_fn)
model.encoder4.register_full_backward_hook(hook_backward_fn)
model.decoder2.register_full_backward_hook(hook_backward_fn)
model.decoder3.register_full_backward_hook(hook_backward_fn)
model.decoder4.register_full_backward_hook(hook_backward_fn)
model.decoder5.register_full_backward_hook(hook_backward_fn)
model.out.register_full_backward_hook(hook_backward_fn)


model.half().to(device)

criterion = nn.HuberLoss()
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters())
# model.train()


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
    print(">>>>>> sliceBatchX: mean ", np.mean(sliceBatchX), " std ", np.std(sliceBatchX))
    print(">>>>>> sliceBatchY: mean ", np.mean(sliceBatchY), " std ", np.std(sliceBatchY))
    inputBatchX[cntBatch, :, :, :, :] = sliceBatchX
    inputBatchY[cntBatch, :, :, :, :] = sliceBatchY
    cntBatch += 1

    if cntBatch == batch_size:
        print(">>>>>> inputBatchX: mean ", np.mean(inputBatchX), " std ", np.std(inputBatchX))
        print(">>>>>> inputBatchY: mean ", np.mean(inputBatchY), " std ", np.std(inputBatchY))

        realInputX = torch.from_numpy(inputBatchX).half().to(device)
        realInputY = torch.from_numpy(inputBatchX).half().to(device)
        realOutput = model(realInputX)
        print("==>Output shape: ", realOutput.size())
        optimizer.zero_grad()
        loss = criterion(realOutput, realInputY)
        loss.backward()
        optimizer.step()

        # print("@"*60)
        # print(model.Parameters())
        # print(model.)
        # print("@"*60)


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
        print("&"*60)



# try to input it to the UNETR model
# H*W*D*C -> patch (N*N*N)
# L = H*W*D*C / N^3


# inputPET = torch.from_numpy(normPET).half().to(device)
# outputPET = model(inputPET)
# print(outputPET.size())
# model.add_module("linear", nn.Linear(in_features = opt.block_size, 
#                                      out_features = opt.block_size)).to(device)

# try to get the output
