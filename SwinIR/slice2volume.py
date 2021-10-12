# python main_test_swinir.py --task classical_sr --scale 2 --training_patch_size 64 --model_path model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth --folder_lq testsets/Set5/LR_bicubic/X2 --folder_gt testsets/Set5/HR

# results/swinir_classical_sr_x2
# -- Average PSNR/SSIM(RGB): 21.81 dB; 0.7678
# -- Average PSNR_Y/SSIM_Y: 23.14 dB; 0.8944

# 3dresample -dxyz 1.367 1.367 1.367 -prefix CUB_011.nii.gz -input CT_011.nii.gz
import nibabel as nib
import numpy as np
import os

ratio = 2
file_sCT = nib.load("./brain/brain_4x.nii.gz")
data_sCT = file_sCT.get_fdata()[:, :, :]
hx, hy, hz = data_sCT.shape
recon = np.zeros(data_sCT.shape)
for idx in range(hz):
    img = np.load("./results/swinir_real_sr_x4_large/brain_{:03d}_SwinIR.npy".format(idx))
    recon[:, :, idx] = img[:, :, 1]

recon = recon * np.amax(data_sCT)
pred_file = nib.Nifti1Image(recon, file_sCT.affine, file_sCT.header)
pred_name = "./brain/REC_brain_4x.nii.gz"
nib.save(pred_file, pred_name)
