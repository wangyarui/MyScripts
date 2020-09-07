import cv2
import numpy as np
import pickle as pickle
import os

file = '/home/osk/datasets/cifar-10-python/cifar-10-python/cifar-10-batches-py/test_batch'

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict


dict1 = unpickle(file)
len_dict = len(dict1[b"filenames"])
for i in range(len_dict):
    img = dict1[b"data"][i]#得到图片的数据
    img = np.reshape(img, (3, 32,32))  #转为三维图片数组
    img = img.transpose((1,2,0))#通道转换为CV2的要求形式
    img_name = dict1[b"filenames"][i].decode()#拿到图片的名字
    img_label = str(dict1[b"labels"][i])#拿到图片的标签

    save_path = os.path.join("/home/osk/datasets/cifar-10-python/cifar-10-dataset/",img_label)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # test = save_path + "/"+img_label+"_"+img_name

    cv2.imwrite(save_path + "/"+img_label+"_"+img_name,img)#保存
