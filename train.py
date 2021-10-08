import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import Net
from data import DatasetFromFolder

from monai.networks.nets import UNet

import sys

def train_a_epoch(data_loader, model, epoch, device, loss_batch_cnt, bp=True):

    epoch_loss = np.zeros((len(data_loader)))
    loss_batch = np.zeros((loss_batch_cnt))
    for iteration, batch in enumerate(data_loader, 0): # start from 0
        batch_x, batch_y = batch[0].to(device), batch[1].to(device)
        # batch_x = torch.from_numpy(batch_x).double()
        # batch_y = torch.from_numpy(batch_y).double()
        # print(batch_x.size())
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        if bp:
            loss.backward()
            optimizer.step()
        loss_voxel = loss.item() / opt.block_size ** 3
        loss_batch[iteration % loss_batch_cnt] = loss_voxel
        epoch_loss[iteration] = loss_voxel

        if iteration % loss_batch_cnt == loss_batch_cnt - 1:
            loss_mean = np.mean(loss_batch)
            loss_std = np.std(loss_batch)
            print("===> Epoch[{}]({}/{}): ".format(epoch + 1, iteration + 1, len(data_loader)), end='')
            print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))

    return epoch_loss


# training setting
parser = argparse.ArgumentParser(description='Use 3d Unet to translate NAC PET to CT')
parser.add_argument('--batch_size', type=int, default=10, help='training batch size')
parser.add_argument('--batch_size_val', type=int, default=10, help='validation batch size')
parser.add_argument('--loss_batch_cnt', type=int, default=53, help='loss display batch')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate. Default=0.01')
parser.add_argument('--data_worker', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=813, help='random seed to use.')
parser.add_argument('--cpu', action='store_true', help='use cuda?')
parser.add_argument('--block_size', type=int, default=128, help='the block size of each input')
parser.add_argument('--stride', type=int, default=64, help='the stride in dataset')
parser.add_argument('--depth', type=int, default=4, help='the depth of unet')
parser.add_argument('--num_filters', type=int, default=8, help='the number of starting filters')
parser.add_argument('--model_tag', type=str, default="XYZ111_Huber_modelv2", help='tag of the current model')
parser.add_argument('--old_model', type=str, default="XYZ111_Huber_modelv2", help='name of the pre-trained model')
parser.add_argument('--continue_train', action='store_true', help='continue training?')
opt = parser.parse_args()
print(opt)

# basic setting
torch.manual_seed(opt.seed)
device = torch.device("cuda" if not opt.cpu else "cpu")
print("Device: ", device)

# set the dataset
trainFolderX = "./data_train/X"+str(opt.block_size)+"/train/"
trainFolderY = "./data_train/Y"+str(opt.block_size)+"/train/"
testFolderX = "./data_train/X"+str(opt.block_size)+"/test/"
testFolderY = "./data_train/Y"+str(opt.block_size)+"/test/"
valFolderX = "./data_train/X"+str(opt.block_size)+"/val/"
valFolderY = "./data_train/Y"+str(opt.block_size)+"/val/"
model_save_path = "model_best_{}.pth".format(opt.old_model)

dataset_train = DatasetFromFolder(data_dir_X = trainFolderX,
                                  data_dir_Y = trainFolderY,
                                  batch_size = 1)
dataset_val = DatasetFromFolder(data_dir_X = valFolderX,
                                data_dir_Y = valFolderY,
                                batch_size = 1)

dataloader_train = DataLoader(dataset=dataset_train,
                              num_workers=opt.data_worker,
                              batch_size=opt.batch_size,
                              shuffle=True)

dataloader_val = DataLoader(dataset=dataset_val,
                            num_workers=opt.data_worker,
                            batch_size=opt.batch_size_val,
                            shuffle=True)
print("===> Datasets and Dataloders are set")

# build the network
# model = Net(block_size = opt.block_size,
#             num_filters = opt.num_filters,
#             num_level = opt.depth,
#             verbose = False).to(device)
# model.double()
# model_save_path
criterion = nn.HuberLoss()

if opt.continue_train:
    model = torch.load(model_save_path).to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("The model has been loaded from: ", model_save_path)

    val_loss = np.asarray(train_a_epoch(dataloader_val, model, 0, 
                                        device, opt.loss_batch_cnt, bp=False))
    val_mean = np.mean(val_loss)
    val_std = np.std(val_loss)
    # np.save("val_{}_{}.npy".format(epoch, opt.model_tag), val_loss)
    print("===> Previous Val Loss, Avg: {:.6}, Std: {:.6}".format(val_mean, val_std))
    val_loss_best = val_mean

else:
    model = UNet(dimensions=3,
                 in_channels=1,
                 out_channels=1,
                 channels=(16, 32, 64, 128, 256),
                 strides=(2, 2, 2, 2, 2),
                 num_res_units=2)
    model.add_module("linear", nn.Linear(in_features = opt.block_size, 
                                         out_features = opt.block_size))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    model.float()
    print("The model has created.")
    val_loss_best = 1e6

# criterion = nn.MSELoss()
# input = torch.randn(4, 1, opt.block_size, opt.block_size, opt.block_size).double().to(device)
# model.summary(input)

# save model architecture
stdoutOrigin=sys.stdout 
sys.stdout = open(opt.model_tag+"_arch.txt", "w")
print(model)
sys.stdout.close()
sys.stdout=stdoutOrigin

print("===> The network, loss, optimizer are set")

# start the training

for epoch in range(opt.epochs):

    epoch_loss = train_a_epoch(dataloader_train, model, epoch, device, opt.loss_batch_cnt)
    epoch_mean = np.mean(epoch_loss)
    epoch_std = np.std(epoch_loss)
    np.save("epoch_Loss_{}_{}.npy".format(epoch, opt.model_tag), epoch_std)
    print("===> Epoch {} Complete Loss, Avg: {:.6}, Std: {:.6}".format(epoch+1, epoch_mean, epoch_std))

    val_loss = train_a_epoch(dataloader_val, model, epoch, device, opt.loss_batch_cnt)
    val_mean = np.mean(val_loss)
    val_std = np.std(val_loss)
    np.save("val_{}_{}.npy".format(epoch, opt.model_tag), val_loss)
    print("===> Val {} Complete Loss, Avg: {:.6}, Std: {:.6}".format(epoch+1, val_mean, val_std))

    if val_mean < val_loss_best:
        torch.save(model, model_save_path)
        print("Checkpoint saved to {}".format(model_save_path))
        val_loss_best = val_mean

# # (N,C,D,H,W)
# input = torch.randn(4, 1, 32, 32, 32).double()
# model.summary(input)


