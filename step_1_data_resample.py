import os
import glob

fileList = glob.glob("./*.nii.gz")
fileList.sort()

for filePath in fileList:
	# print(filePath)
	fileName = os.path.basename(filePath)
	# CT_011.nii.gz
	fileIndex = fileName[-10:-7]
	# print(fileIndex)
	cmd = "3dresample -dxyz 1.367 1.367 1.367 -prefix CUB_"+fileIndex+".nii.gz -input NPR_"+fileIndex+".nii.gz"
	print(cmd)
	os.system(cmd)