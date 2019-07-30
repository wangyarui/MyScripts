# 批量修改文件夹下的文件名
import os

path = '/home/osk/图片/ocr/bad'

# 获取该目录下所有文件，存入列表中
fileList = os.listdir(path)
fileList.sort()

n = 0
for i in fileList:
    # 设置旧文件名（就是路径+文件名）
    oldname = path + os.sep + fileList[n]  # os.sep添加系统分隔符

    # 设置新文件名
    number = str(n).zfill(3)
    newname = path  + os.sep + 'bad_' + str(number) + '.png'

    os.rename(oldname, newname)  # 用os模块中的rename方法对文件改名
    print(oldname, '======>', newname)

    n += 1
