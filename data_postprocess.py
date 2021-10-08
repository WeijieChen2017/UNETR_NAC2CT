from PIL import Image
import nibabel as nib
import numpy as np
import glob
import os

def create_index_3d(data, block_size, stride):
    
    data_size = data.shape
    pad_width = []
    for len_dim in data_size:
        before_pad_width = (len_dim - len_dim // block_size * block_size) // 2
        after_pad_width = len_dim - len_dim // block_size * block_size - before_pad_width
        pad_width.append((before_pad_width, after_pad_width))
    data_pad = np.pad(data, pad_width, mode = "constant")
    
    list_start = []
    for len_dim in data_pad.shape:
        list_dim = []
        max_start = (len_dim - block_size) // stride
        for idx in range(max_start + 1):
            list_dim.append((idx * stride, idx * stride + block_size))
        list_start.append(list_dim)
    
    return list_start, data_pad

def remove_pad(data_pad, data_ori, block_size, stride):
    data_size = data_ori.shape
    pad_width = []
    for len_dim in data_size:
        before_pad_width = (len_dim - len_dim // block_size * block_size) // 2
        after_pad_width = len_dim - len_dim // block_size * block_size - before_pad_width
        pad_width.append((before_pad_width, after_pad_width))

    before_x, after_x = pad_width[0]
    before_y, after_y = pad_width[1]
    before_z, after_z = pad_width[2]

    data_cut = data_pad[before_x:-after_x, before_y:-after_y, before_z:-after_z]
    if data_cut.shape[0] == 0:
        data_cut = data_pad[:, :, before_z:-after_z]

    print("Data_cut shape: ", data_cut.shape)
    np.save("data_cut.npy", data_cut)
    print("Saved")
    return denormY(data_cut)

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

folderX = "./data_train/NPR_SRC/"
folderY = "./data_train/CT_SRC/"
valRatio = 0.2
testRatio = 0.1
channelX = 1
channelY = 1
block_size = 128
stride = 64

# create directory and search nifty files
testFolderX = "./data_train/X128/test/"
testFolderY = "./data_train/Y128/test/"

# for folderName in [testFolderX, testFolderY]:
#     if not os.path.exists(folderName):
#         os.makedirs(folderName)

# testList = glob.glob(testFolderX+"/*.nii") + glob.glob(testFolderY+"/*.nii.gz")
# testList.sort()
# for testPath in testList:
#     print(testPath)

# shuffle and create train/val/test file list
np.random.seed(813)
testList = ['./data_train/RSPET/RS_011.nii.gz']
            # './data_train/NPR_SRC/NPR_063.nii.gz',
            # './data_train/NPR_SRC/NPR_143.nii.gz']
folder_pred_cube = "./data_pred/cube/"
folder_pred_nifty = "./data_pred/nifty/"

print('-'*50)
print("Testing list: ", testList)
print('-'*50)

packageTest = [testList, testFolderX, testFolderY, "Test"]

for package in [packageTest]:
    fileList = package[0]
    folderX = package[1]
    folderY = package[2]
    print("-"*25, package[3], "-"*25)
    
    # npy version
    for pathX in fileList:
        pathY = pathX.replace("NPR", "CT")
        filenameX = os.path.basename(pathX)[3:6]
        filenameY = os.path.basename(pathY)[3:6]
        fileX = nib.load(pathX)
        fileY = nib.load(pathY)
        dataX = fileX.get_fdata()
        dataY = fileY.get_fdata()
        dataNormX = normX(dataX)
        dataNormY = normY(dataY)
        print("Input shape: ", dataNormX.shape, dataNormY.shape)

        # listStart, dataPadX = create_index_3d(dataNormX, block_size, stride)
        # listStart, dataPadY = create_index_3d(dataNormY, block_size, stride)
        # data_pred = np.zeros(dataPadX.shape)

        # listCordX = listStart[0]
        # listCordY = listStart[1]
        # listCordZ = listStart[2]

        # for start_x, end_x in listCordX:
        #     for start_y, end_y in listCordY:
        #         for start_z, end_z in listCordZ:
        #             savename = folder_pred_cube + "pred_" + filenameX 
        #             savename += "_{0:03d}_{1:03d}_{2:03d}".format(start_x, start_y, start_z) + ".npy"
        #             # print(savename)
        #             cube_pred = np.load(savename)
        #             data_pred[start_x:end_x, start_y:end_y, start_z:end_z] = cube_pred

        # data_cut = remove_pad(data_pred, dataNormX, block_size, stride)
        data_cut = denormY(np.load("data_cut.npy"))

        data_dif = dataY - data_cut
        pred_file = nib.Nifti1Image(data_cut, fileX.affine, fileY.header)
        diff_file = nib.Nifti1Image(data_dif, fileX.affine, fileX.header)
        pred_name = folder_pred_nifty+"pred_"+filenameX+".nii.gz"
        diff_name = folder_pred_nifty+"diff_"+filenameX+".nii.gz"
        nib.save(pred_file, pred_name)
        nib.save(diff_file, diff_name)

        print("&"*10)
        print(pred_name)
