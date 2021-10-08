import numpy as np
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import Net
from data import DatasetFromFolder

# training setting

parser = argparse.ArgumentParser(description='Use 3d Unet to translate NAC PET to CT')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--loss_batch_cnt', type=int, default=100, help='eval loss batch size')
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
testFolderX = "./data_train/X"+str(opt.block_size)+"/test/"
testFolderY = "./data_train/Y"+str(opt.block_size)+"/test/"
testSaveFolder = "./data_pred/cube"
niftySaveFolder = "./data_pred/nifty"

for folder_name in [testSaveFolder, niftySaveFolder]:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

dataset_test = DatasetFromFolder(data_dir_X = testFolderX,
                                 data_dir_Y = testFolderY,
                                 batch_size = 1,
                                 filename = True)

dataloader_test = DataLoader(dataset=dataset_test,
                             num_workers=opt.data_worker,
                             batch_size=opt.batch_size,
                             shuffle=True)

print("===> Datasets and Dataloders are set")

print("Model name: ", opt.model_save_path)
criterion = nn.HuberLoss()
model = torch.load(opt.model_save_path).to(device)
model.eval()
print("===> The model {} are loaded.".format(opt.model_save_path))

epoch_loss = np.zeros((len(dataloader_test)))
loss_batch = np.zeros((opt.loss_batch_cnt))
n_samples = len(dataloader_test)
for iteration, batch in enumerate(dataloader_test, 0):
    batch_x, batch_y, filename = batch[0].to(device), batch[1].to(device), batch[2]
    sample_name = os.path.basename(filename[0][0])
    sample_path = os.path.join(testSaveFolder, sample_name.replace("X", "pred"))
    pred = model(batch_x)
    np.save(sample_path, np.squeeze(pred.detach().numpy()))
    loss = criterion(pred, batch_y).item() / opt.block_size ** 3
    
    loss_batch[iteration % opt.loss_batch_cnt] = loss
    epoch_loss[iteration] = loss

    if iteration % opt.loss_batch_cnt == opt.loss_batch_cnt - 1:
        loss_mean = np.mean(loss_batch)
        loss_std = np.std(loss_batch)
        print("===> Eval({}/{}): ".format(iteration + 1, n_samples), end='')
        print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))

print("The eval loss mean: ", np.mean(epoch_loss), " std: ", np.std(epoch_loss))
np.save("eval_loss.npy", epoch_loss)

