import os
import os.path
from PIL import Image

# 准备拉伸的原图片存储路径
infile = r'C:\Users\12859\Desktop\123\123\dog_gray'
# 拉伸后的图片存储路径
outfile = r'C:\Users\12859\Desktop\123\123\dog_gray_2828'

list_img = os.listdir(infile)
n = 0
l = len(list_img)
for each_img in list_img:
    # 每个图像全路径
    print(each_img)
    image_input_fullname = infile + '/' + each_img
    resize_img = Image.open(image_input_fullname)
    # 可以自定义大小
    out_w = 28
    out_h = 28
    out_img = resize_img.resize((out_w, out_h), Image.ANTIALIAS)

    # 裁剪后每个图像的路径+名称
    image_output_fullname = outfile + "/" + each_img
    out_img.convert('RGB')
    out_img.save(image_output_fullname)
    n += 1

    print('%d/%d img has been resized!' % (n, l))

print('total_num is {%d} success resized img!' % len(list_img))
