import os
import glob

fileList = glob.glob("./*.nii.gz")
fileList.sort()

for filePath in fileList:
	print(filePath)
	fileName = os.path.basename(filePath)
	# CT_011.nii.gz
	fileIndex = fileName[-10:-7]
	print(fileIndex)