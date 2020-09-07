# 异常图片生成器
import os
import random
from torchvision import transforms
from PIL import Image
import cv2

def get_files(path, rule):
    all = []
    for fpathe,dirs,fs in os.walk(path):
        for f in fs:
            filename = os.path.join(fpathe,f)
            if filename.endswith(rule):
                all.append(filename)
    return all

image_transforms = transforms.Compose([
    # transforms.RandomErasing(),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),

])

transform_toimage = transforms.Compose([
    transforms.ToPILImage()
])

if __name__ == '__main__':

    img_path = '/home/osk/datasets/cifar-10-python/ganomaly/car_truck/train/0/'
    img_list = get_files(img_path,'.png')

    sample_image_list = random.sample(img_list,10)

    for samle_image_path in sample_image_list:
        samle_image = Image.open(samle_image_path)
        # img = Image.fromarray(samle_image)

        img = image_transforms(samle_image)

        show_img = transform_toimage(img)

        _,base_name = os.path.split(samle_image_path)
        savepath = '/home/osk/datasets/cifar-10-python/ganomaly/car_truck/test/1/'+ 'fake_' + base_name


        show_img.save(savepath)
