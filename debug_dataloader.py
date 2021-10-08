import argparse
import numpy as np
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
parser.add_argument('--test_batch_size', type=int, default=4, help='testing batch size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--data_worker', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=813, help='random seed to use.')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--block_size', type=int, default=32, help='the block size of each input')
parser.add_argument('--stride', type=int, default=32, help='the stride in dataset')
parser.add_argument('--depth', type=int, default=2, help='the depth of unet')
parser.add_argument('--num_filters', type=int, default=16, help='the number of starting filters')
opt = parser.parse_args()
print(opt)

# basic setting
torch.manual_seed(opt.seed)
device = torch.device("cuda" if opt.cuda else "cpu")

# set the dataset
trainFolderX = "./data_train/X/train/"
trainFolderY = "./data_train/Y/train/"
testFolderX = "./data_train/X/test/"
testFolderY = "./data_train/Y/test/"
valFolderX = "./data_train/X/val/"
valFolderY = "./data_train/Y/val/"

dataset_train = DatasetFromFolder(data_dir_X = trainFolderX,
                                  data_dir_Y = trainFolderY,
                                  batch_size = 1,
                                  filename=True)
dataset_val = DatasetFromFolder(data_dir_X = valFolderX,
                                data_dir_Y = valFolderY,
                                batch_size = 1)

dataloader_train = DataLoader(dataset=dataset_train,
                              num_workers=opt.data_worker,
                              batch_size=opt.batch_size,
                              shuffle=True)

dataloader_val = DataLoader(dataset=dataset_val,
                            num_workers=opt.data_worker,
                            batch_size=opt.batch_size,
                            shuffle=True)
print("===> Datasets and Dataloders are set")

# build the network
model = Net(block_size = opt.block_size,
            num_filters = opt.num_filters,
            num_level = opt.depth).to(device)
model.double()
criterion = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
print("===> The network, loss, optimizer are set")

# start the training

for epoch in range(opt.epochs):

    epoch_loss = 0
    cnt =0
    for iteration, batch in enumerate(dataloader_train, 1):
        batch_x, batch_y = batch[0].to(device), batch[1].to(device)
        sample_name = os.path.basename(batch[2][0])
        np.save("./"+sample_name, batch_x)
        np.save("./"+sample_name.replace("X", "Y"), batch_y)
        cnt += 1
        if cnt == 10:
            exit()    

        # batch_x = torch.from_numpy(batch_x).double()
        # batch_y = torch.from_numpy(batch_y).double()
        # print(batch_x.type())
        # optimizer.zero_grad()
        # loss = criterion(model(batch_x), batch_y)
        # epoch_loss += loss.item()
        # loss.backward()
        # optimizer.step()

    #     print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(dataloader_train), loss.item()))

    # print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(dataloader_train)))

    # model_save_path = "model_epoch_{}.pth".format(epoch)
    # torch.save(model, model_save_path)
    # print("Checkpoint saved to {}".format(model_save_path))

# # (N,C,D,H,W)
# input = torch.randn(4, 1, 64, 64, 64).double()
# model.summary(input)


