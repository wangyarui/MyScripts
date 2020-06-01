import cv2
import numpy as np

# YUV直方图均衡化
def YUVequal(src_img):
    imgYUV = cv2.cvtColor(src_img, cv2.COLOR_BGR2YCrCb)
    cv2.imshow("src", src_img)

    channelsYUV = cv2.split(imgYUV)
    channelsYUV[0] = cv2.equalizeHist(channelsYUV[0])

    channels = cv2.merge(channelsYUV)
    result = cv2.cvtColor(channels, cv2.COLOR_YCrCb2BGR)
    cv2.namedWindow("dst",cv2.WINDOW_NORMAL)
    cv2.imshow("dst", result)
    cv2.waitKey(0)

# RGB图像均衡化
def RGBequal(src_image):
    # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
    (b, g, r) = cv2.split(src_image)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # 合并每一个通道
    result = cv2.merge((bH, gH, rH))
    cv2.namedWindow("dst",cv2.WINDOW_NORMAL)
    cv2.imshow("dst", result)

    cv2.waitKey(0)

'''图像伪彩色增强—实现：https://zhuanlan.zhihu.com/p/59992663'''
def colorAugment(src_image):
    result = cv2.applyColorMap(src_image,cv2.COLORMAP_HOT)
    cv2.namedWindow("dst", cv2.WINDOW_NORMAL)
    cv2.imshow("dst", result)

    cv2.waitKey(0)

'''多对象匹配'''
def multi_match_template(src_image,template_image):
    img_gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    h, w = template_image.shape

    res = cv2.matchTemplate(img_gray, template_image, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    # 取匹配程度大于%80的坐标
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):  # *号表示可选参数
        bottom_right = (pt[0] + w, pt[1] + h)
        cv2.rectangle(src_image, pt, bottom_right, (0, 255, 0), 2)

    cv2.namedWindow("img_rgb",cv2.WINDOW_NORMAL)
    cv2.imshow('img_rgb', src_image)
    cv2.waitKey(0)





