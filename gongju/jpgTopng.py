import os
import cv2

def renameFile(img_Dir):
        img_pathDir = os.listdir(img_Dir)                           # 提取所有文件名，并存在列表中
        print(img_Dir)                                              # 输出文件路径
        print(img_pathDir)                                          # 输出文件名列表
        print(len(img_pathDir))                                     # 输出文件数
        for i in range(len(img_pathDir)):
            img_name = img_pathDir[i]                               # 变量储存：文件名+拓展名
            img_path_name = img_Dir + img_name                      # 变量储存：绝对路径+文件名+拓展名
            print(img_name)                                         # 输出文件名+拓展名
            print(img_path_name)                                    # 输出绝对路径+文件名+拓展名
            # img_new_path_name = img_path_name[:-4] + '.png'       # 只更改文件的拓展名，通常只改为png，会导致图片读不出来
            # print(img_new_path_name)
            img = cv2.imread(img_path_name)
            number_name = img_Dir+str(i+1) + '.png'                 # 改拓展名和文件名，文件名为序号，从1开始
            print(number_name)
            cv2.imwrite(number_name, img)                           # 用opencv读一下文件，再存出来，改名为png，但原文件还在
            os.remove(img_path_name)                                # 删除原文件
            # os.rename(img_path_name, number_name)                 # 直接用os的rename命令改名
        return

if __name__ == '__main__':
    img_Dir = "C:/Users/Administrator/Desktop/456/test/0"
    renameFile(img_Dir)



