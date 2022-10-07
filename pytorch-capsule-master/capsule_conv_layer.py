import torch.nn as nn

# 编码器中的卷积层，输入(128,1,,28,28)的图片，输出(128,256,20,20)的张量
class CapsuleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CapsuleConvLayer, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=9, # fixme constant
                               stride=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)
        # inplace = True 时,会修改输入对象的值,所以打印出对象存储地址相同,类似于C语言的址传递
        # 节省反复申请与释放内存的空间与时间,只是将原来的地址传递,效率更好

    def forward(self, x):
        return self.relu(self.conv0(x))


