import nibabel as nib
import numpy as np
import glob
import os

def merge_block(blockSeq):
    # return np.mean(blockSeq, axis=0)
    return np.median(blockSeq, axis=0)

def denormY(data):
    return data * 4000 - 1000

def remove_pad(dataPad, dataSize, blockSize, stride):
    padWidth = []
    for lenDim in dataSize:
        beforePadWidth = (lenDim - lenDim // blockSize * blockSize) // 2
        afterPadWidth = lenDim - lenDim // blockSize * blockSize - beforePadWidth
        padWidth.append((beforePadWidth, afterPadWidth))

    beforeX, afterX = padWidth[0]
    beforeY, afterY = padWidth[1]
    beforeZ, afterZ = padWidth[2]

    dataCut = dataPad[beforeX:-afterX, beforeY:-afterY, beforeZ:-afterZ]
    if dataCut.shape[0] == 0:
        dataCut = dataPad[:, :, beforeZ:-afterZ]
    return denormY(dataCut)


pathCube = "./data_pred/cube/"
pathAssm = "./data_pred/assm/"
filePred = nib.load("./data_train/RSPET/RS_011.nii.gz")
dataPred = filePred.get_fdata()
dataSize = dataPred.shape
blockSize = 128
stride = 32
predAssm = np.zeros(dataSize)
overlapCnt = blockSize // stride
print(dataSize, overlapCnt)
if not os.path.exists(pathAssm):
        os.makedirs(pathAssm)

listStart = []
for lenDim in dataSize:
    listCube = []
    maxStart = (lenDim - blockSize) // stride
    for idx in range(maxStart + 1):
        listCube.append(idx * stride)
    listStart.append(listCube)
    print("-> Starting coordinates: ", listCube)

numCube = len(listStart[0])*len(listStart[1])*len(listStart[2])
cntCube = 0
for iX in range(len(listStart[0])):
    for iY in range(len(listStart[1])):
        for iZ in range(len(listStart[2])):
            assmX, assmY, assmZ = iX*stride, iY*stride, iZ*stride
            
            # generate the relative cube list
            choice = []
            for i in [iX, iY, iZ]:
                subChoice = []
                for off in range(overlapCnt):
                    s = i + off-overlapCnt+1
                    if s >= 0:
                        subChoice.append(s)
                choice.append(subChoice)
            print("Choice of {}-{}-{} for three directions: ".format(iX, iY, iZ), choice)
            cubeSeq = np.zeros((len(choice[0])*len(choice[1])*len(choice[2]), stride, stride, stride))
            
            # load them into a sequence
            cnt = 0
            for sX in choice[0]:
                for sY in choice[1]:
                    for sZ in choice[2]:
                        cordX, cordY, cordZ = listStart[0][sX], listStart[1][sY], listStart[2][sZ]
                        eX, eY, eZ = iX - sX, iY - sY, iZ - sZ
                        cutX = cordX + eX*stride
                        cutY = cordY + eY*stride
                        cutZ = cordZ + eZ*stride
                        dataBlock = np.load(pathCube + "pred_011_{0:03d}_{1:03d}_{2:03d}.npy".format(cordX, cordY, cordZ))
                        cubeSeq[cnt, :, :, :] = dataBlock[eX * stride : (eX+1) * stride,
                                                          eY * stride : (eY+1) * stride,
                                                          eZ * stride : (eZ+1) * stride]
            # assembly the seq
            predAssm[assmX:assmX+stride, assmY:assmY+stride, assmZ:assmZ+stride] = merge_block(cubeSeq)
            print("==> Finish[{}]/[{}]: ".format(cntCube, numCube), assmX, assmY, assmZ)
            cntCube += 1

# dataCut = np.load("predAssm.npy")
np.save("predAssm.npy", predAssm)
dataCut = denormY(predAssm)
# dataCut = remove_pad(predAssm, dataSize, blockSize, stride)
dataDiff = dataPred - dataCut
predFile = nib.Nifti1Image(dataCut, filePred.affine, filePred.header)
diffFile = nib.Nifti1Image(dataDiff, filePred.affine, filePred.header)
nib.save(predFile, pathAssm+"/Assm_011.nii.gz")
nib.save(diffFile, pathAssm+"/Vary_011.nii.gz")
cmdCut = "3dresample -dxyz 1.367 1.367 3.27 -prefix pred_011.nii.gz -input Assm_011.nii.gz"
cmdDif = "3dresample -dxyz 1.367 1.367 3.27 -prefix diff_011.nii.gz -input Vary_011.nii.gz"
os.system("cd ./data_pred/assm/")
os.system(cmdCut)
os.system(cmdDif)


