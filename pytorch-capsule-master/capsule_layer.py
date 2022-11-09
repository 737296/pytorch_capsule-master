# 按照是否需要动态路由(use_routing)完成编码器中的primary层和digits层，并且完成挤压的函数定义

'''
CapsuleConvLayer:卷积层， in (128,1,,28,28)，out (128,256,20,20)

primary:
1. 初始化：一个unit为stride=2的(128,256,9,9)卷积，in (128,256,20,20) out (128, 32, 6, 6)
          一个self.units为8个unit
    操作： 将数据x带入胶囊进行卷积，得到(128,8,32,6,6),reshape成(128,8,1152),在对1152这个数据维度进行squash操作，返回(128,8,1152)
           最后得到的结果为(128,8,1152)

digits:
1. 初始化：初始化了一个可以计算梯度的参数W,w.shape=(1, 1152, 10, 16, 8)
    操作：进行动态路由更新c值
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# 接收解码器中卷积层的in (128,256,20,20) 进行stride=2的(128,256,9,9)卷积 out (128, 32, 6, 6)
class ConvUnit(nn.Module):
    def __init__(self, in_channels):
        super(ConvUnit, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=in_channels,   # in_channels=256
                               out_channels=32,
                               kernel_size=9,
                               stride=2,
                               bias=True)

    def forward(self, x):
        return self.conv0(x)

# primary层和digits层
class CapsuleLayer(nn.Module):
    def __init__(self, in_units, in_channels, num_units, unit_size, use_routing):
        super(CapsuleLayer, self).__init__()

        self.in_units = in_units
        self.in_channels = in_channels
        self.num_units = num_units
        self.use_routing = use_routing

        # 在初始化中定义
        # digits层使用
        if self.use_routing:
            # In the paper, the deeper capsule layer(s) with capsule inputs (DigitCaps) use a special routing algorithm
            # that uses this weight matrix.
            # in_channels=32*32*6,num_units=10, unit_size=16, in_units=8
            self.W = nn.Parameter(torch.randn(1, in_channels, num_units, unit_size, in_units))
            # self.W.shape=torch.Size([1, 1152, 10, 16, 8])
            # randn函数，形成一个这么多维度的标准正态分布

        # primary层使用
        else:
            # The first convolutional capsule layer (PrimaryCapsules in the paper) does not perform routing.
            # The first convolutional capsule layer (PrimaryCapsules in the paper) does not perform routing.
            # Instead, it is composed of several convolutional units, each of which sees the full input.
            # It is implemented as a normal convolutional layer with a special nonlinearity (squash()).

            def create_conv_unit(unit_idx):
                unit = ConvUnit(in_channels=in_channels)   # in_channels=256
                self.add_module("unit_" + str(unit_idx), unit)   # 在网络架构中增加 unit0……
                return unit
            # 一个unit为stride=2的(128,256,9,9)卷积，in (128,256,20,20) out (128, 32, 6, 6)

            self.units = [create_conv_unit(i) for i in range(self.num_units)]   # self.num_units=8
            # self.units为8个unit(注意依旧是方法，还未调用）

    # 无需实例化，直接类名.方法名调用
    @staticmethod
    def squash(s):
        mag_sq = torch.sum(s**2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, x):
        if self.use_routing:
            return self.routing(x)
        else:
            return self.no_routing(x)

    # primary层
    def no_routing(self, x):
        # 该层是 PrimaryCaps 层，能组合第 1 层检测到的低级特征。该层由32个胶囊组成(理论，但是在代码中没有看到）(u为一个胶囊)
        # 每个胶囊又含有8个步长为 2 的 9×9×256 卷积核，输入为 20×20×256 的张量，胶囊输出为 6×6×8×32 的张量。
        # self.units为 8个 unit = stride为2的(128,256,9,9)卷积，in (128,256,20,20) out (128, 32, 6, 6)
        # num_units=8,x.shape=torch.Size([128, 256, 20, 20])
        # self.units[i](x)为迭代8次调用卷积，每次返回torch.Size([128, 32, 6, 6])，并将结果保存在一个列表中
        u = [self.units[i](x) for i in range(self.num_units)]  # len(u)=8, u[0].data.shape=torch.Size([128, 32, 6, 6])

        # Stack all unit outputs (batch, unit, channels, height, width).
        u = torch.stack(u, dim=1)   # 对所有unit进行叠加,u.shape=torch.Size([128, 8, 32, 6, 6])

        # Flatten to (batch, unit, output).
        u = u.view(x.size(0), self.num_units, -1)  # num_units=8,u.shape=torch.Size([128, 8, 1152])  1152=32*6*6

        # Return squashed outputs.
        # 仅对 1152 这一个维度进行压缩操作
        return CapsuleLayer.squash(u)   # (128,8,1152)

    # digits层，接收primary层的 (128,8,1152),输出 torch.Size([128, 10, 16, 1])
    def routing(self, x):
        batch_size = x.size(0)  # x.size(0)=128  x.size(1)=8  x.size(2)=1152

        # (batch, in_units, features) -> (batch, features, in_units)
        x = x.transpose(1, 2)   # 从(128,8,1152)变成t(128,1152,8)

        # (batch, features, in_units) -> (batch, features, num_units, in_units, 1)
        x = torch.stack([x] * self.num_units, dim=2).unsqueeze(4)

        # (batch, features, in_units, unit_size, num_units)
        W = torch.cat([self.W] * batch_size, dim=0)

        # Transform inputs by weight matrix.
        # (batch_size, features, num_units, unit_size, 1)
        # 仿射变换
        u_hat = torch.matmul(W, x)

        # Initialize routing logits to zero.
        # b_ij = torch.zeros(1, self.in_channels, self.num_units, 1).cpu()
        b_ij = torch.zeros(1, self.in_channels, self.num_units, 1)

        # Iterative routing.
        num_iterations = 3
        for iteration in range(num_iterations):
            # Convert routing logits to softmax.
            # (batch, features, num_units, 1, 1)
            c_ij = F.softmax(b_ij,dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            # Apply routing (c_ij) to weighted inputs (u_hat).
            # (batch_size, 1, num_units, unit_size, 1)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            # (batch_size, 1, num_units, unit_size, 1)
            v_j = CapsuleLayer.squash(s_j)

            # (batch_size, features, num_units, unit_size, 1)
            v_j1 = torch.cat([v_j] * self.in_channels, dim=1)

            # (1, features, num_units, 1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)

            # Update b_ij (routing)
            b_ij = b_ij + u_vj1
        return v_j.squeeze(1)
