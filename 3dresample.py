import os
import glob

os.system("set AFNI_NIFTI_TYPE_WARN = NO")
# ct_list = glob.glob("./CT_SRC/CT_*.nii.gz").sort()
# pet_list = glob.glob("./NP_SRC/NP_*.nii.gz").sort()
file_list = glob.glob("./*.nii.gz").sort()

for file_path in file_list:

    file_name = os.path.basename(file_path)
    case_num = file_name[3:6]
    print(case_num)
    cmd_1 = "3dresample -dxyz 1 1 1 -prefix RS_"+case_num+".nii.gz -input "+file_name+" -rmode Cu"
    for cmd in [cmd_1]:
        print(cmd)
        os.system(cmd)




    # cmd_1 = "3dresample -prefix NPR_"+name+".nii.gz -master CT_"+name+".nii.gz -rmode Cu -input NP_"+name+".nii.gz -bound_type SLAB"
    # cmd_2 = "3dZeropad -I 17 -S 17 p"+idx_str+"+orig"
    # cmd_3 = "3dAFNItoNIFTI -prefix "+tag+idx_str+" "+tag+idx_str+"+orig"
    # cmd_4 = "rm -f zeropad+orig.BRIK"
    # cmd_5 = "rm -f zeropad+orig.HEAD"
    # cmd_6 = "rm -f "+tag+idx_str+"+orig.BRIK"
    # cmd_7 = "rm -f "+tag+idx_str+"+orig.HEAD"
    # cmd_8 = "mv "+tag+idx_str+".nii ./PET_2x/"
    # cmd_6 = "mv y"+idx_str+".nii ../inv_RSZP"

# 3dresample -dxyz 1.367 1.367 3.27 -prefix pred_011_RS.nii.gz -input pred_011.nii.gz
# 3dZeropad -I 16 -S 17 -A 25 -P 26 -L 25 -R 26 Z001+orig -prefix 123
# 3dAFNItoNIFTI -prefix test test+orig