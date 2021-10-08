from PIL import Image
import nibabel as nib
import numpy as np
import glob
import os

from multiprocessing import Pool, cpu_count
# import affinity
# os.system("taskset -p 0xff %d" % os.getpid())
os.sched_setaffinity(0, range(2**cpu_count()-1))
# affinity.set_process_affinity_mask(0, 2**multiprocessing.cpu_count()-1)

def save_each_nifty(folderX, folderY, pathX):

    pathY = pathX.replace("NPR", "CT")
    filenameX = os.path.basename(pathX)[4:7]
    filenameY = os.path.basename(pathY)[3:6]
    dataX = nib.load(pathX).get_fdata()
    dataY = nib.load(pathY).get_fdata()
    dataNormX = normX(dataX)
    dataNormY = normY(dataY)

    listStart, dataPadX = create_index_3d(dataNormX, block_size, stride)
    listStart, dataPadY = create_index_3d(dataNormY, block_size, stride)
    
    listCordX = listStart[0]
    listCordY = listStart[1]
    listCordZ = listStart[2]

    for start_x, end_x in listCordX:
        for start_y, end_y in listCordY:
            for start_z, end_z in listCordZ:
                savenameX = folderX + "X_" + filenameX 
                savenameX += "_{0:03d}_{1:03d}_{2:03d}".format(start_x, start_y, start_z) + ".npy"
                savenameY = folderY + "Y_" + filenameY
                savenameY += "_{0:03d}_{1:03d}_{2:03d}".format(start_x, start_y, start_z) + ".npy"
                np.save(savenameX, dataPadX[start_x:end_x, start_y:end_y, start_z:end_z])
                np.save(savenameY, dataPadY[start_x:end_x, start_y:end_y, start_z:end_z])
    print("&"*10)
    print(filenameX)
    print(len(listCordX) * len(listCordY) * len(listCordZ), " files are saved.")

    return os.getpid()

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

def normX(data):
    data[data<0] = 0
    data[data>3000] = 3000  
    data = data / 3000
    return data

def normY(data):
    data[data<-1000] = -1000
    data[data>3000] = 3000
    data = (data + 1000) / 4000
    return data

folderX = "./data_train/NPR_SRC/"
folderY = "./data_train/CT_SRC/"
valRatio = 0.2
testRatio = 0.1
channelX = 1
channelY = 1
block_size = 64
stride = 16

# create directory and search nifty files
trainFolderX = "./data_train/X"+str(block_size)+"/train/"
trainFolderY = "./data_train/Y"+str(block_size)+"/train/"
testFolderX = "./data_train/X"+str(block_size)+"/test/"
testFolderY = "./data_train/Y"+str(block_size)+"/test/"
valFolderX = "./data_train/X"+str(block_size)+"/val/"
valFolderY = "./data_train/Y"+str(block_size)+"/val/"

for folderName in [trainFolderX, testFolderX, valFolderX,
                   trainFolderY, testFolderY, valFolderY]:
    if not os.path.exists(folderName):
        os.makedirs(folderName)

fileList = glob.glob(folderX+"/*.nii") + glob.glob(folderX+"/*.nii.gz")
fileList.sort()
for filePath in fileList:
    print(filePath)

# shuffle and create train/val/test file list
np.random.seed(813)
fileList = np.asarray(fileList)
np.random.shuffle(fileList)
fileList = list(fileList)

valList = fileList[:int(len(fileList)*valRatio)]
valList.sort()
testList = fileList[-int(len(fileList)*testRatio):]
testList.sort()
trainList = list(set(fileList) - set(valList) - set(testList))
trainList.sort()

# trainList = ['./data_train/NPR_SRC/NPR_051.nii.gz',
#              './data_train/NPR_SRC/NPR_054.nii.gz',
#              './data_train/NPR_SRC/NPR_056.nii.gz',
#              './data_train/NPR_SRC/NPR_057.nii.gz']
# valList = ['./data_train/NPR_SRC/NPR_059.nii.gz']
# testList = ['./data_train/NPR_SRC/NPR_011.nii.gz']

# --------------------------------------------------
# Training list:  ['./data_train/NPR_SRC/NPR_001.nii.gz', './data_train/NPR_SRC/NPR_007.nii.gz', './data_train/NPR_SRC/NPR_017.nii.gz', './data_train/NPR_SRC/NPR_019.nii.gz', './data_train/NPR_SRC/NPR_024.nii.gz', './data_train/NPR_SRC/NPR_026.nii.gz', './data_train/NPR_SRC/NPR_028.nii.gz', './data_train/NPR_SRC/NPR_029.nii.gz', './data_train/NPR_SRC/NPR_031.nii.gz', './data_train/NPR_SRC/NPR_044.nii.gz', './data_train/NPR_SRC/NPR_057.nii.gz', './data_train/NPR_SRC/NPR_059.nii.gz', './data_train/NPR_SRC/NPR_067.nii.gz', './data_train/NPR_SRC/NPR_068.nii.gz', './data_train/NPR_SRC/NPR_078.nii.gz', './data_train/NPR_SRC/NPR_082.nii.gz', './data_train/NPR_SRC/NPR_095.nii.gz', './data_train/NPR_SRC/NPR_098.nii.gz', './data_train/NPR_SRC/NPR_101.nii.gz', './data_train/NPR_SRC/NPR_103.nii.gz', './data_train/NPR_SRC/NPR_104.nii.gz', './data_train/NPR_SRC/NPR_130.nii.gz', './data_train/NPR_SRC/NPR_138.nii.gz', './data_train/NPR_SRC/NPR_142.nii.gz', './data_train/NPR_SRC/NPR_159.nii.gz']
# --------------------------------------------------
# Validation list:  ['./data_train/NPR_SRC/NPR_051.nii.gz', './data_train/NPR_SRC/NPR_054.nii.gz', './data_train/NPR_SRC/NPR_056.nii.gz', './data_train/NPR_SRC/NPR_097.nii.gz', './data_train/NPR_SRC/NPR_127.nii.gz', './data_train/NPR_SRC/NPR_128.nii.gz', './data_train/NPR_SRC/NPR_133.nii.gz']
# --------------------------------------------------
# Testing list:  ['./data_train/NPR_SRC/NPR_011.nii.gz', './data_train/NPR_SRC/NPR_063.nii.gz', './data_train/NPR_SRC/NPR_143.nii.gz']
# --------------------------------------------------


print('-'*50)
print("Training list: ", trainList)
print('-'*50)
print("Validation list: ", valList)
print('-'*50)
print("Testing list: ", testList)
print('-'*50)

packageTrain = [trainList, trainFolderX, trainFolderY, "Train"]
packageVal = [valList, valFolderX, valFolderY, "Validation"]
packageTest = [testList, testFolderX, testFolderY, "Test"]
np.save("dataset_division.npy", [packageTrain, packageVal, packageTest])

dataLoaderPool = Pool()

for package in [packageTest, packageVal, packageTrain]:

    fileList = package[0]
    folderX = package[1]
    folderY = package[2]
    print("-"*25, package[3], "-"*25)

    # npy version
    for pathX in fileList:
        print(pathX, ' '*4, end='')
        pid = dataLoaderPool.apply_async(save_each_nifty, args=(folderX, folderY, pathX)).get()
        print("==>Current PID: ", pid, "finished. ")

dataLoaderPool.close()
dataLoaderPool.join()


    