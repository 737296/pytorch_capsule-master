# 完成编码器的定义，前向传播算法，损失函数的定义
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.utils as vutils
from capsule_conv_layer import CapsuleConvLayer
from capsule_layer import CapsuleLayer


class CapsuleNetwork(nn.Module):
    def __init__(self,
                 image_width,  # 28
                 image_height,  # 28
                 image_channels,  # 1
                 conv_inputs,  # 1
                 conv_outputs,  # 256
                 num_primary_units,  # 8
                 primary_unit_size,  # 32*32*6
                 num_output_units,  # 10  one for each MNIST digit
                 output_unit_size):  # 16
        super(CapsuleNetwork, self).__init__()  # 进行nn.Module中的初始化，但是nn.Module没有初始化任何参数

        self.reconstructed_image_count = 0

        self.image_channels = image_channels
        self.image_width = image_width
        self.image_height = image_height

        # 定义编码器中的卷积层:conv1，进行一个卷积然后relu输出
        self.conv1 = CapsuleConvLayer(in_channels=conv_inputs,  # 1,表示的是通道数
                                      out_channels=conv_outputs)  # 256\

        # 定义编码器中的primary层:primary
        self.primary = CapsuleLayer(in_units=0,
                                    in_channels=conv_outputs,  # 256
                                    num_units=num_primary_units,  # 8
                                    unit_size=primary_unit_size,  # 32*32*6
                                    use_routing=False)

        # 定义编码器中的digits层:digits
        self.digits = CapsuleLayer(in_units=num_primary_units,  # 8
                                   in_channels=primary_unit_size,  # 32*32*6
                                   num_units=num_output_units,  # 10
                                   unit_size=output_unit_size,  # 16
                                   use_routing=True)

        reconstruction_size = image_width * image_height * image_channels  # 28*28*1
        #
        self.reconstruct0 = nn.Linear(num_output_units * output_unit_size, int((reconstruction_size * 2) / 3))
        self.reconstruct1 = nn.Linear(int((reconstruction_size * 2) / 3), int((reconstruction_size * 3) / 2))
        self.reconstruct2 = nn.Linear(int((reconstruction_size * 3) / 2), reconstruction_size)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    # 重写父类中的forward函数，该名字不可以修改
    def forward(self, x):
        # print(x.shape)   torch.Size([128, 1, 28, 28]) 可以知道就是Dataloader中的data
        return self.digits(self.primary(self.conv1(x)))

    # 不是重写，在main函数中被调用
    def loss(self, images, input, target, size_average=True):
        return self.margin_loss(input, target, size_average) + self.reconstruction_loss(images, input, size_average)

    # margin_loss，在loss中被调用
    def margin_loss(self, input, target, size_average=True):
        batch_size = input.size(0)

        # ||vc|| from the paper.
        v_mag = torch.sqrt((input ** 2).sum(dim=2, keepdim=True))

        # Calculate left and right max() terms from equation 4 in the paper.
        # zero = Variable(torch.zeros(1)).cpu()
        zero = Variable(torch.zeros(1))
        m_plus = 0.9
        m_minus = 0.1
        max_l = torch.max(m_plus - v_mag, zero).view(batch_size, -1) ** 2
        max_r = torch.max(v_mag - m_minus, zero).view(batch_size, -1) ** 2

        # This is equation 4 from the paper.
        loss_lambda = 0.5
        T_c = target
        L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
        L_c = L_c.sum(dim=1)

        if size_average:
            L_c = L_c.mean()

        return L_c

    # reconstruction_loss，在loss中被调用
    def reconstruction_loss(self, images, input, size_average=True):
        # Get the lengths of capsule outputs.
        v_mag = torch.sqrt((input ** 2).sum(dim=2))

        # Get index of longest capsule output.
        _, v_max_index = v_mag.max(dim=1)
        v_max_index = v_max_index.data

        # Use just the winning capsule's representation (and zeros for other capsules) to reconstruct input image.
        batch_size = input.size(0)
        all_masked = [None] * batch_size
        for batch_idx in range(batch_size):
            # Get one sample from the batch.
            input_batch = input[batch_idx]

            # Copy only the maximum capsule index from this batch sample.
            # This masks out (leaves as zero) the other capsules in this sample.
            batch_masked = Variable(torch.zeros(input_batch.size()))
            # batch_masked = Variable(torch.zeros(input_batch.size())).cpu()
            batch_masked[v_max_index[batch_idx]] = input_batch[v_max_index[batch_idx]]
            all_masked[batch_idx] = batch_masked

        # Stack masked capsules over the batch dimension.
        masked = torch.stack(all_masked, dim=0)

        # Reconstruct input image.
        masked = masked.view(input.size(0), -1)
        output = self.relu(self.reconstruct0(masked))
        output = self.relu(self.reconstruct1(output))
        output = self.sigmoid(self.reconstruct2(output))
        output = output.view(-1, self.image_channels, self.image_height, self.image_width)

        # Save reconstructed images occasionally.
        if self.reconstructed_image_count % 10 == 0:
            if output.size(1) == 2:
                # handle two-channel images
                zeros = torch.zeros(output.size(0), 1, output.size(2), output.size(3))
                output_image = torch.cat([zeros, output.data], dim=1)
                # output_image = torch.cat([zeros, output.data.cpu()], dim=1)
            else:
                # assume RGB or grayscale
                # output_image = output.data.cpu()
                output_image = output.data
            vutils.save_image(output_image, "reconstruction.png")
        self.reconstructed_image_count += 1

        # The reconstruction loss is the sum squared difference between the input image and reconstructed image.
        # Multiplied by a small number so it doesn't dominate the margin (class) loss.
        error = (output - images).view(output.size(0), -1)
        error = error ** 2
        error = torch.sum(error, dim=1) * 0.0005

        # Average over batch
        if size_average:
            error = error.mean()

        return error
