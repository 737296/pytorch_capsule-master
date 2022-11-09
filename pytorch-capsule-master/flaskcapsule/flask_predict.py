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
#��ȡ��ǰ·��
basedir = os.path.abspath(os.path.dirname(__file__))

@app.route('/api/upload', methods=['POST'])
def uploadImage():
    img=request.files.get("photos")
    path=basedir+"\\predictimage\\"
    file_path = path + img.filename
    #����ԭʼͼƬ
    img.save(file_path)
    #��������ͼƬ
    out_image=imageTo2828AndRGBTOL(file_path)
    out_image.save(file_path)
    #��ʼԤ��
    reslut=predict(file_path)
    return jsonify({"code": 200, "message": "success","data":reslut})


def predict(file_path):
    # ������ҪԤ���ͼƬ(�Ѿ���������)
    img_pth = file_path
    img = Image.open(img_pth)
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(img)
    image = torch.reshape(image, (1, 1, 28, 28))
    # ��������
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

    # ѡ��Ȩ���ļ�
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
        return "������"
    if pred.item()==1:
        return "��������"
    if pred.item()==2:
        return "ɳ������"

def imageTo2828AndRGBTOL(file_path):
    resize_img = Image.open(file_path)
    # �����Զ����С
    out_w = 28
    out_h = 28
    out_img = resize_img.resize((out_w, out_h), Image.ANTIALIAS)
    out_img.convert('RGB')
    #�Ҷ�
    out_img=out_img.convert('L')
    return out_img

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
