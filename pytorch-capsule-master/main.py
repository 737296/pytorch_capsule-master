import datetime
import torch
import torch.optim as optim
from torchvision import datasets, transforms

from capsule_network import CapsuleNetwork

# 记录开始时间
time_start = datetime.datetime.now()
# log存储
time = datetime.datetime.now().strftime('%Y-%m-%d,%H-%M-%S')
flieName = '../log/' + time + '.txt'
file = open(flieName, 'a')
file.writelines('准确度\n')
file.close()

#
# Settings.
#

learning_rate = 0.01

batch_size = 8
test_batch_size = 8

# Stop training if loss goes below this threshold.
early_stop_loss = 0.0001

#
# Load MNIST dataset.
#

# Normalization for MNIST dataset.
dataset_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('../data', train=True, download=True, transform=dataset_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST('../data', train=False, download=True, transform=dataset_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

#
# Create capsule network.
#

conv_inputs = 1
conv_outputs = 256
num_primary_units = 8
primary_unit_size = 32 * 6 * 6  # fixme get from conv2d
output_unit_size = 16

# 初始化胶囊网络参数，在初始化时，首先会观察CapsuleNetwork中的__init__,
# 当其中包括如CapsuleLayer时，会先跳入执行CapsuleLayer的__init__,再返回来执行CapsuleNetwork的__init__.
network = CapsuleNetwork(image_width=28,
                         image_height=28,
                         image_channels=1,
                         conv_inputs=conv_inputs,
                         conv_outputs=conv_outputs,
                         num_primary_units=num_primary_units,
                         primary_unit_size=primary_unit_size,
                         num_output_units=3,  # one for each MNIST digit
                         output_unit_size=output_unit_size).cuda()


# 打印网络结构
# print(network)

# 展示network中需要计算的参数
# print(network.parameters())
# for group in network.parameters():
#    print(group.shape)   # 每一个group中的参数是一层卷积或者一层胶囊中的参数大小

# Converts batches of class indices to classes of one-hot vectors.
def to_one_hot(x, length):
    batch_size = x.size(0)
    x_one_hot = torch.zeros(batch_size, length)
    for i in range(batch_size):
        x_one_hot[i, x[i]] = 1.0
    return x_one_hot


# This is the test function from the basic Pytorch MNIST example, but adapted to use the capsule network.
def test(epoch):
    network.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        target_indices = target
        target_one_hot = to_one_hot(target_indices, length=network.digits.num_units)

        data, target = data.cuda(), target_one_hot.cuda()

        output = network(data)

        test_loss += network.loss(data, output, target, size_average=False).data[0]  # sum up batch loss

        v_mag = torch.sqrt((output ** 2).sum(dim=2, keepdim=True))

        pred = v_mag.data.max(1, keepdim=True)[1].cpu()

        correct += pred.eq(target_indices.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    # 向log写入准确度
    i = round((correct / len(test_loader.dataset) * 100).item(), 2)
    file = open(flieName, 'a')
    file.writelines(str(i) + "\n")
    file.close()
    # 记录结束时间
    time_end = datetime.datetime.now()
    time111=(time_end-time_start).seconds
    print("time:" + str(time111) + "s")
    print('Test Epoch:{},Average loss: {:.4f}, Accuracy: ({}/{}),({:.2%})'.format(
        epoch,
        test_loss,
        correct,
        len(test_loader.dataset),
        correct / len(test_loader.dataset)),
    )



def train_start(epoch):
    # network.parameters()在网络中通过nn.Parameter来定义
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    last_loss = None
    log_interval = 1

    # 训练批次batch_idx：0
    # 图像数据data：torch.Size([128, 1, 28, 28])
    # 结果target：torch.Size([128])
    for batch_idx, (data, target) in enumerate(train_loader):
        target_one_hot = to_one_hot(target, length=network.digits.num_units)

        data, target = data.cuda(), target_one_hot.cuda()  # 使用cuda核心进行训练，以前是Variable(data).cuda()

        optimizer.zero_grad()  # 将梯度归零

        output = network(data)  # 在将类对象当作函数调用时，会自动调用__call__方法中的forward函数，然后卷积层，primary层，digits层

        loss = network.loss(data, output, target)  # 计算损失

        # 反向传播计算得到每个参数的梯度值
        loss.backward()  # 所以虽然在胶囊层改变了连接方式，但是其实还是按照反向传播的方式更新梯度

        last_loss = loss.data.item()

        optimizer.step()  # 通过梯度下降执行一步参数更新

        # if batch_idx % log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch,
        #         batch_idx * len(data),
        #         len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader),
        #         loss.data.item()))

        if last_loss < early_stop_loss:
            break

    return last_loss


def start(num_epochs):
    for epoch in range(1, num_epochs + 1):
        last_loss = train_start(epoch)  # 执行一次训练
        test(epoch)  # 执行一次测试
        if last_loss < early_stop_loss:
            break


def log():
    # time = datetime.datetime.now().strftime('%Y-%m-%d,%H-%M-%S')
    # flieName = '../log/' + time + '.txt'
    # file = open(flieName, 'w')
    # file.write('准确度')
    print('{:.10%}'.format(907564 / 1008467))
    print(907564 / 1008467 * 100)


if __name__ == "__main__":
    num_epochs = 50
    start(num_epochs)
    # 写入log 数据
    file = open(flieName, 'a')
    file.writelines("epochs:" + str(num_epochs) + "\n")
    time_end = datetime.datetime.now()
    file.writelines("time:" + str((time_end - time_start).seconds) + "\n")
    file.close()
