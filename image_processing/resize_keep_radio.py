#!/usr/bin/python
#coding:utf-8

'''resize图片且保持原始的图片长宽比尺寸'''
import cv2

def resize_img_keep_ratio(img,target_size):
    old_size= img.shape[0:2]
    #ratio = min(float(target_size)/(old_size))
    # test = len(old_size)
    ratio = min(float(target_size[i])/(old_size[i]) for i in range(len(old_size)))
    new_size = tuple([int(i*ratio) for i in old_size])
    img = cv2.resize(img,(new_size[1], new_size[0]))
    pad_w = target_size[1] - new_size[1]
    pad_h = target_size[0] - new_size[0]
    top,bottom = pad_h//2, pad_h-(pad_h//2)
    left,right = pad_w//2, pad_w -(pad_w//2)
    img_new = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,None,(0,0,0))

    cv2.imshow('test',img_new)
    cv2.waitKey(0)
    return img_new

if __name__ == '__main__':
    src_img = cv2.imread("./images/test_1.bmp",0)

    et, thre = cv2.threshold(src_img, 127, 255, cv2.THRESH_OTSU)

    target_size = [80, 80]
    img_new = resize_img_keep_ratio(thre, target_size)