# python main_test_swinir.py --task classical_sr --scale 2 --training_patch_size 64 --model_path model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth --folder_lq testsets/Set5/LR_bicubic/X2 --folder_gt testsets/Set5/HR

# 3dresample -dxyz 1.367 1.367 1.367 -prefix CUB_011.nii.gz -input CT_011.nii.gz
import nibabel as nib
import numpy as np

def normY(data):
    data[data<-1000] = -1000
    data[data>3000] = 3000
    data = (data + 1000) / 4000
    return data

def get_index(current_idx, max_idx):
    if current_idx == 0:
        return [0, 0, 1]
    if current_idx == max_idx:
        return [max_idx-1, max_idx, max_idx]
    else:
        return [current_idx-1, current_idx, current_idx+1]

def volume2slice(data, save_folder):
    dx, dy, dz = data.shape
    img = np.zeros((dx, dy, 3))
    for idx in range(dz):
        idx_set = get_index(idx, dz-1)
        img[:, :, 0] = data[:, :, idx_set[0]]
        img[:, :, 1] = data[:, :, idx_set[1]]
        img[:, :, 2] = data[:, :, idx_set[2]]
        np.save(save_folder+"_CT011_{:3d}.npy".format(idx), img)
        print("Save imgs in "+save_folder+" [{:3d}]/[{:3d}]".format(idx+1, dz+1))


# (imgname, imgext) = os.path.splitext(os.path.basename(path))
# img_gt = np.load(path)
# img_lq = np.load(f'{args.folder_lq}/{imgname}x{args.scale}{imgext}')

ratio = 4
file_sCT = nib.load("./dataset/sctTr/CUB_011.nii.gz")
data_sCT = file_sCT.get_fdata()[:, :, :]
norm_sCT = normY(data_sCT)
hx, hy, hz = norm_sCT.shape
lx, ly, lz = hx//ratio, hy//ratio, hz
resz_sCT = np.resize(norm_sCT, (lx, ly, lz))
print(norm_sCT.shape, resz_sCT)

volume2slice(norm_sCT, "./test/CT/HR/")
volume2slice(resz_sCT, "./test/CT/LR/")
