import os
from glob import glob
import random
from dataset_processing.utils.kh_tools import *
import matplotlib.pyplot as plt
import cv2


def get_files(path, rule):
    all = []
    for fpathe,dirs,fs in os.walk(path):
        for f in fs:
            filename = os.path.join(fpathe,f)
            # if filename.endswith(('.jpg', '.bmp', '.png')):
            if filename.endswith(rule):
                all.append(filename)
    return all

if __name__ == '__main__':

    img_list = get_files('/home/osk/0_work/0_Denso/dataset/mvtec_ad/bottle/test/1', ('.png', '.jpg'))

    for img_path in img_list:

        path_,img = os.path.split(img_path)

        basename,_ = os.path.split(path_)

        prefix = basename.split('/')[-2]

        new_name = path_ + '/' + prefix + '_' + img

        os.rename(img_path,new_name)


        print('test')
