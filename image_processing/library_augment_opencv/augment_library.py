import cv2

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