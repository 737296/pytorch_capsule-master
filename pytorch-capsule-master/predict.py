# coding=gbk
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
from capsule_network import CapsuleNetwork

# 导入需要预测的图片(已经经过处理)
img_pth = "./predictImg/P_ (19).png"
img = Image.open(img_pth)
transform = transforms.Compose([transforms.ToTensor()])
image=transform(img)

image = torch.reshape(image, (1, 1, 28, 28))
# 加载网络
device = torch.device("cuda")
image=image.to(device)

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
                         output_unit_size=output_unit_size).cuda()

# 选择权重文件
model = torch.load("./model/crackAndDamageAndNormal.99.pth", map_location=device)
network.load_state_dict(model)
network.eval()
print(image.shape)
print(image)
out = network(image)
v_mag = torch.sqrt((out ** 2).sum(dim=2, keepdim=True))
print(v_mag)
pred = v_mag.data.max(1, keepdim=True)[1].cuda()
print(pred)

# out = F.softmax(out, dim=1)
# out = out.data.cpu().numpy()
# print(out)
