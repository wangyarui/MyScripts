import os
import cv2

save_path = "/home/osk/datasets/SCUT-FORU/SCUT_FORU_DB_Release/English2k/SCUT_seg_character/"


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

    fp = open('./scut_foru_dataset.txt', 'w+')
    img_list = get_files('/home/osk/datasets/SCUT-FORU/SCUT_FORU_DB_Release/English2k/character_img/', ('.bmp', '.jpg'))

    for img in img_list:
        img_src = cv2.imread(img)

        dirname,basename = os.path.split(img)
        # filename,_ = os.path.splitext(basename)

        labels_path =   img.replace("jpg","txt").replace("character_img","character_annotation")
        txtFile = open(labels_path)
        txtLines = txtFile.readlines()
        print(labels_path)

        for i in range(len(txtLines)):
            if i == 0:
                continue

            context = txtLines[i].strip('\n').split(',')
            print(context)

            x0 = int(context[0])
            y0 = int(context[1])
            x1 = int(context[0]) + int(context[2])
            y1 = int(context[1]) + int(context[3])

            # if context[4] == "0":
            #     context[4] = "O"

            anno = context[4].replace('"', ' ')#.upper()

            fp.write(str(i) + "_" + basename + " " + anno + '\n')

            cv2.imwrite(save_path + str(i) + "_" + basename, img_src[y0:y1,x0:x1] )
            # print('test')
    fp.close()




        # for singLine in txtLines:
        #     singLine = singLine.strip('\n')
        #     for i in range(len(singLine)):
        #         filename = singLine.split(' ')[i]
        #
        #
        #         print('test')
# txtFile = open(root[1])
# txtLines = txtFile.readlines()
#
# for singLine in txtLines:
#     singLine = singLine.strip('\n')
#     filename = singLine.split(' ')[0]
#     filepath = os.path.join('/home/osk/datasets/OMSdatatset/oms_dataset/street_view_image', filename)