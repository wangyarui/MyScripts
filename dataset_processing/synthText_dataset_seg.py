from scipy.io import loadmat
import os
from tqdm import tqdm
import cv2
import numpy as np


gt_mat_path = "/home/osk/datasets/SynthText/SynthText/gt.mat"
images_path = "/home/osk/datasets/SynthText/SynthText/"
txt_path = "/home/osk/datasets/SynthText/SynthText/gt/"

data = loadmat(gt_mat_path)

files,txt,wordBB,charBB= data['imnames'],data['txt'],data['wordBB'],data['charBB']

for i, imname in enumerate(tqdm(data['imnames'][0])):
    imname = imname[0]
    img_id = os.path.basename(imname)
    img_name,_ = os.path.splitext(img_id)
    im_path = os.path.join(images_path, imname)

    img=cv2.imread(im_path)

    char_list = []
    char_list = ''.join([str(w)for w in txt[0,i]])
    char_list = ''.join(char_list.split())

    char_BB = charBB[0][i]

    top_left_x = np.asarray(char_BB[0][0])
    top_right_x = np.asarray(char_BB[0][1])
    bottom_right_x = np.asarray(char_BB[0][2])
    bottom_left_x = np.asarray(char_BB[0][3])
    top_left_y = np.asarray(char_BB[1][0])
    top_right_y = np.asarray(char_BB[1][1])
    bottom_right_y = np.asarray(char_BB[1][2])
    bottom_left_y = np.asarray(char_BB[1][3])


    for j in range(0, len(char_list)):
        pts = np.array([[int(top_left_x[j]),int(top_left_y[j])],[int(top_right_x[j]),int(top_right_y[j])],[int(bottom_right_x[j]),int(bottom_right_y[j])],[int(bottom_left_x[j]),int(bottom_left_y[j])],[int(top_left_x[j]),int(top_left_y[j])]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,(0,0,255),2)
        cv2.putText(img,char_list[j],(int(top_left_x[j]),int(top_left_y[j])),cv2.FONT_HERSHEY_PLAIN,1.25,(0,255,0),2) #BGR-Red

        cv2.imwrite(txt_path + char_list[j] + "_" + str(j) + img_name + ".jpg" , img([pts]))

        print('test')




    # cv2.imshow("test",img)
    # cv2.waitKey(0)
    # cv2.imwrite('/path/to/test_gts_not_polygon3_char.png',img)
# print(data.keys())

# for i, imname in enumerate(tqdm(data['imnames'][0])):
#     imname = imname[0]
#     img_id = os.path.basename(imname)
#     im_path = os.path.join(images_path, imname)
#     txt_path = os.path.join(txt_path, img_id.replace('jpg', 'txt'))
#
#     if len(data['wordBB'][0,i].shape) == 2:
#         annots = data['wordBB'][0,i].transpose(1, 0).reshape(-1, 8)
#     else:
#         annots = data['wordBB'][0,i].transpose(2, 1, 0).reshape(-1, 8)
#     with open(txt_path, 'w') as f:
#         f.write(imname + '\n')
#         for annot in annots:
#             str_write = ','.join(annot.astype(str).tolist())
#             f.write(str_write + '\n')


# single_data = {}

# for i in range(data["imnames"]):


# print(data["charBB"][0][1])
# print(data["imnames"][0][1])
# print(data["txt"][0][1])

# for single_data in data[0]:
#     charBB = single_data["charBB"]
#     imnames = single_data["imnames"]
#     charName = single_data["txt"]
#     print('test')


