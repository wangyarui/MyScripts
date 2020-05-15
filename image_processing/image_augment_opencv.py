from image_processing.library_augment_opencv.augment_library import *

if __name__ == '__main__':
    src_img = cv2.imread('../image_features/images/CaptureImage_0001.png')

    # #YUV直方图均衡化
    # YUVequal(src_img)

    #RGB直方图均衡化
    RGBequal(src_img)