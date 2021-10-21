import os
import glob

# fileList = glob.glob("./*.nii.gz")
# fileList.sort()

# for filePath in fileList:
#     # print(filePath)
#     fileName = os.path.basename(filePath)
#     # CT_011.nii.gz
#     fileIndex = fileName[-10:-7]
#     # print(fileIndex)
#     cmd = "3dresample -dxyz 2 2 1 -prefix RSZ_"+fileIndex+".nii.gz -input CUB_"+fileIndex+".nii.gz"
#     print(cmd)
#     os.system(cmd)

for idx in range(64):
    cmd = "3dresample -dxyz 2 2 1 -prefix RSZ_{:03d}.nii.gz -input MR__MLAC_".format(idx)+str(idx)+"_MNI.nii.gz"
    print(cmd)
    os.system(cmd)