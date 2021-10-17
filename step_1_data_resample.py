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
	cmd = "3dresample -dxyz 5.468 5.468 5.468 -prefix RSZ_"+fileIndex+".nii.gz -input CUB_"+fileIndex+".nii.gz"
	print(cmd)
	os.system(cmd)