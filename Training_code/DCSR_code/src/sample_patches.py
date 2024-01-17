import glob
import os
import random
import shutil
import numpy as np

if __name__ == '__main__':
    num_file_folder = 15
    ratio_sample = 0.20
    names_hr = []
    names_lr = []
    dir_hr = R'E:\QingtangDing\dataset\DIV2K\data_50_cluster_15'
    dir_lr = r'E:\QingtangDing\dataset\DIV2K\train_LR_full\X2'
    dir_hr_save = r'E:\QingtangDing\dataset\DIV2K\train_HR_50_15_resample_20_uniform'
    dir_lr_save = r'E:\QingtangDing\dataset\DIV2K\train_LR_50_15_resample_20_uniform\X2'
   
    total_number = 131165
    # total_number = 426530  # flickr2k
    number_per_cluster = int(total_number * ratio_sample / num_file_folder)
    sample_number = 0
    for i in range(num_file_folder):
        name_hr = sorted(glob.glob(os.path.join(dir_hr, str(i), '*' + ".png")))
        index = random.sample(range(len(name_hr)), number_per_cluster)
        names_hr += np.array(name_hr)[index].tolist()
    for path_hr in names_hr:
        file_name = os.path.basename(path_hr)
        shutil.copy(path_hr, os.path.join(dir_hr_save, file_name))
        file_name_split = file_name.split(".")[0].split("_")
        file_name_lr = file_name_split[0] + "x2_" + file_name_split[-1] + ".png"
        path_lr = os.path.join(dir_lr, file_name_lr)
        shutil.copy(path_lr, os.path.join(dir_lr_save, file_name_lr))
