import nibabel as nib
import numpy as np
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import monai

def normX(data):
    data[data<0] = 0
    data[data>6000] = 6000
    data = data / 6000
    return data

def normY(data):
    data[data<-1000] = -1000
    data[data>3000] = 3000
    data = (data + 1000) / 4000
    return data

def denormX(data):
    return data * 6000

def denormY(data):
    return data * 4000 - 1000

# training setting

parser = argparse.ArgumentParser(description='Use 3d Unet to translate NAC PET to CT')
parser.add_argument('--batch_size', type=int, default=10, help='training batch size')
parser.add_argument('--data_worker', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=813, help='random seed to use.')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--block_size', type=int, default=128, help='the block size of each input')
parser.add_argument('--model_save_path', type=str, default='model_best_XYZ111_Huber_modelv2.pth')

opt = parser.parse_args()
print(opt)

# basic setting
torch.manual_seed(opt.seed)
device = torch.device("cuda" if opt.cuda else "cpu")

# set the dataset
filenameX = "011"
testX = "./data_train/RSPET/RS_011.nii.gz"
testY = "./data_train/RSCT/RS_011.nii.gz"
dirSave = "./data_pred/"

fileX = nib.load(testX)
fileY = nib.load(testY)
dataX = fileX.get_fdata()
dataY = fileY.get_fdata()
inputX = normX(np.expand_dims(dataX, axis=(0, 1)))
print("Shape X: ", dataX.shape, " Shape Y: ", dataY.shape, " Input X: ", inputX.shape)

print("===> Datasets and Dataloders are set")

print("Model name: ", opt.model_save_path)
criterion = nn.HuberLoss()
model = torch.load(opt.model_save_path).float().to(device)
model.eval()
print("===> The model {} are loaded.".format(opt.model_save_path))

pred = monai.inferers.sliding_window_inference(
        inputs=torch.from_numpy(inputX).float().to(device), #NCHW[D]
        roi_size=opt.block_size, 
        sw_batch_size=opt.batch_size, 
        predictor=model, 
        overlap=0.5, 
        mode=monai.utils.BlendMode.CONSTANT, 
        sigma_scale=0.125, # not valid if constant 
        padding_mode=monai.utils.PytorchPadMode.CONSTANT, 
        cval=0.0, 
        sw_device=None, 
        device=None).numpy()

print("The loss is ", criterion(pred, normY(dataY)).item())

data_pred = denormY(pred)
data_diff = dataY - data_pred
pred_file = nib.Nifti1Image(data_pred, fileX.affine, fileX.header)
diff_file = nib.Nifti1Image(data_diff, fileX.affine, fileX.header)
pred_name = dirSave+"pred_"+filenameX+".nii.gz"
diff_name = dirSave+"diff_"+filenameX+".nii.gz"
nib.save(pred_file, pred_name)
nib.save(diff_file, diff_name)

cmd0 = "3dresample -dxyz 1.367 1.367 3.27 -prefix DLCT_011_.nii.gz -input pred_011.nii.gz"
cmd1 = "3dresample -dxyz 1.367 1.367 3.27 -prefix DMAP_011_.nii.gz -input diff_011.nii.gz"

for cmd in [cmd0, cmd1]:
    print(cmd0)
    os.system(cmd0)


# epoch_loss = 0
# epoch_loss_list = []
# for iteration, batch in enumerate(dataloader_test, 1):
#     batch_x, batch_y, filename = batch[0].to(device), batch[1].to(device), batch[2]
#     sample_name = os.path.basename(filename[0][0])
#     sample_path = os.path.join(testSaveFolder, sample_name.replace("X", "pred"))
#     pred = model(batch_x)
#     np.save(sample_path, np.squeeze(pred.detach().numpy()))
#     loss = criterion(pred, batch_y)
#     epoch_loss += loss.item()
#     print("===> ({}/{}): Loss: {:.6f}".format(iteration, len(dataloader_test), loss.item()))
#     epoch_loss_list.append(loss.item())

# print("The loss is ", epoch_loss / len(dataloader_test))
# np.save("val_loss.npy", np.asarray(epoch_loss_list))

