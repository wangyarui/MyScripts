from image_processing.library_augment_opencv.augment_library import *
import time

if __name__ == '__main__':
    src_img = cv2.imread('../image_features/images/CaptureImage_0001.png')
    gray_img = cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)

    # #YUV直方图均衡化
    # YUVequal(src_img)
    # #RGB直方图均衡化
    # RGBequal(src_img)
    # #图像伪彩色增强
    # colorAugment(gray_img)
    #多对象模板匹配
    template_image = cv2.imread('../image_features/images/CaptureImage_00001.png',0)
    time1 = time.time()
    multi_match_template(src_img,template_image)
    inter_time = time.time()-time1
    print(inter_time)
    # template_image2 = cv2.imread('../image_features/images/CaptureImage_00002.png',0)
    # multi_match_template(src_img, template_image2)


