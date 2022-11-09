# -*- coding: GBK -*-
# -*- coding: UTF-8 -*-
# coding=gbk
import os
from flask import jsonify, request
from flask import Flask
# coding=gbk
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
from capsule_network import CapsuleNetwork
app = Flask(__name__)
#获取当前路径
basedir = os.path.abspath(os.path.dirname(__file__))

@app.route('/api/upload', methods=['POST'])
def uploadImage():
    img=request.files.get("photos")
    path=basedir+"\\predictimage\\"
    file_path = path + img.filename
    #保存原始图片
    img.save(file_path)
    #重新缩放图片
    out_image=imageTo2828AndRGBTOL(file_path)
    out_image.save(file_path)
    #开始预测
    reslut=predict(file_path)
    return jsonify({"code": 200, "message": "success","data":reslut})


def predict(file_path):
    # 导入需要预测的图片(已经经过处理)
    img_pth = file_path
    img = Image.open(img_pth)
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(img)
    image = torch.reshape(image, (1, 1, 28, 28))
    # 加载网络
    device = torch.device("cpu")

    conv_inputs = 1
    conv_outputs = 256
    num_primary_units = 8
    primary_unit_size = 32 * 6 * 6  # fixme get from conv2d
    output_unit_size = 16
    network = CapsuleNetwork(image_width=28,
                             image_height=28,
                             image_channels=1,
                             conv_inputs=conv_inputs,
                             conv_outputs=conv_outputs,
                             num_primary_units=num_primary_units,
                             primary_unit_size=primary_unit_size,
                             num_output_units=3,  # one for each MNIST digit
                             output_unit_size=output_unit_size)

    # 选择权重文件
    model = torch.load("../model/crackAndDamageAndNormal.99.pth", map_location=device)
    network.load_state_dict(model)
    network.eval()
    # print(image.shape)
    # print(image)
    out = network(image)
    v_mag = torch.sqrt((out ** 2).sum(dim=2, keepdim=True))
    pred = v_mag.data.max(1, keepdim=True)[1]
    # out = F.softmax(out, dim=1)
    # print(v_mag)
    print(pred)
    if pred.item()==0:
        return "无损伤"
    if pred.item()==1:
        return "裂纹损伤"
    if pred.item()==2:
        return "沙眼损伤"

def imageTo2828AndRGBTOL(file_path):
    resize_img = Image.open(file_path)
    # 可以自定义大小
    out_w = 28
    out_h = 28
    out_img = resize_img.resize((out_w, out_h), Image.ANTIALIAS)
    out_img.convert('RGB')
    #灰度
    out_img=out_img.convert('L')
    return out_img

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
