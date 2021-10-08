import nibabel as nib
import numpy as np

import glob
import os

def normX(data):
    data[data<0] = 0
    data[data>3000] = 6000  
    data = data / 6000
    return data

# try to load one data
filePET = nib.load("./dataset/petTr/RS_011.nii.gz")
dataPET = filePET.get_fdata()
normPET = normX(dataPET)
print(normPET.shape)

# try to input it to the UNETR model
# try to get the output
