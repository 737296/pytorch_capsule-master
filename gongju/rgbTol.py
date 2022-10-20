from PIL import Image
import os

INPUT_PATH = r'C:\Users\Administrator\Desktop\456\train\2' #原始图像存储路径
OUPUT_PATH = r'C:\Users\Administrator\Desktop\456\train_gray\2' #转化为灰度图像 存储路径
files_list = os.listdir(INPUT_PATH)  # 读取列表信息，可打印查看

for file in files_list:
    # Version1
    I = Image.open(INPUT_PATH + "/" + file)
    L = I.convert('L')	# Image包内将“L“代表灰度
    L.save(OUPUT_PATH + "/" + file)
