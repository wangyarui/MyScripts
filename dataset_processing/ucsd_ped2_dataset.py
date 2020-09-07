import os
from glob import glob
import random
from dataset_processing.utils.kh_tools import *
import matplotlib.pyplot as plt

ped2_data_path_train = '/home/osk/datasets/ped2/training/frames'

lst_image_paths = []


for s_image_dir_path in glob(os.path.join(ped2_data_path_train, '*')):
    for sImageDirFiles in glob(os.path.join(s_image_dir_path+'/*')):
        lst_image_paths.append(sImageDirFiles)

lst_forced_fetch_data = [lst_image_paths[x] for x in random.sample(range(0, len(lst_image_paths)), 10)]

# sample_num = 128
# path_size = (10,10)

sample_files = lst_forced_fetch_data[0:128]
patch_size = (10,10)
patch_step = (1,1)
b_work_on_patch = True
train_size = 4

sample,_ = read_lst_images(sample_files, patch_size, patch_step, b_work_on_patch)

sample = np.array(sample).reshape(-1, patch_size[0], patch_size[1], 1)
sample = sample[0:128]

sample_inputs = np.array(sample).astype(np.float32)
scipy.misc.imsave('./test/train_input_samples.jpg', montage(sample_inputs[:,:,:,0]))

# batch_idxs = min(len(sample),4)


print('test')