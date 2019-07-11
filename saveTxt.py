#coding:utf-8
#读取文件目录,保存到txt文件中
#并再次读出,保存到list中
 
import os
 
fp = open('./img_name.txt','w+')
Img_list = os.listdir('/home/osk/datasets/coco_2017/train2017/train2017')
for Name in Img_list:
    # fp.write(str) 将str写到文件中,并不会在str后加上换行符
    fp.write(Name + '\n')
 
#以上,读取目录,并保存(写)到txt文件......
 
 
#fp = open('./img_name.txt','r+')   #注意模式
#for i in range(len(Img_list)):
#    print fp.readline()
#以上,读取一行.....
 
 
fp = open('./img_name.txt','r+')   #不再次打开,new_list为空
new_list = fp.readlines()
print(new_list)
 
new2_list = []
for i in new_list:
    new2_list.append(i[:-1])    #去掉含有的'\n'
print(new2_list)
 
#以上,返回目录list,并去掉'\n'保存到new2_list中......
 
 
fp.close()

