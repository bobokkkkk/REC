import os
import torch.nn.functional as F
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3, out_ch=1, **kwargs):
        super(U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out


class U_Net_no_residual(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3, out_ch=1, **kwargs):
        super(U_Net_no_residual, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[3], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[2], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[1], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[0], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out


class U_Net_single(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3, out_ch=1, **kwargs):
        super(U_Net_single, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[0], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        d2 = self.Up2(e2)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out


class U_Net_residual_c(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3, out_ch=1, **kwargs):
        super(U_Net_residual_c, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        if not "conn_layer" in kwargs:
            self.conn_layer = [0, 1, 2, 3]
        else:
            self.conn_layer = kwargs["conn_layer"]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        if 3 in self.conn_layer:
            self.Up_conv5 = conv_block(filters[4], filters[3])
        else:
            self.Up_conv5 = conv_block(filters[3], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        if 2 in self.conn_layer:
            self.Up_conv4 = conv_block(filters[3], filters[2])
        else:
            self.Up_conv4 = conv_block(filters[2], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        if 1 in self.conn_layer:
            self.Up_conv3 = conv_block(filters[2], filters[1])
        else:
            self.Up_conv3 = conv_block(filters[1], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        if 0 in self.conn_layer:
            self.Up_conv2 = conv_block(filters[1], filters[0])
        else:
            self.Up_conv2 = conv_block(filters[0], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        if 3 in self.conn_layer:
            d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        if 2 in self.conn_layer:
            d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        if 1 in self.conn_layer:
            d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        if 0 in self.conn_layer:
            d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out

class seq_conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(seq_conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv(x)
        return x


class seq_up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(seq_up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
class Sequel_U_net(nn.Module):

    def __init__(self, in_ch=1024, out_ch=1024, **kwargs):
        super(Sequel_U_net, self).__init__()
        n1 = 32
        filters = [n1 * 16, n1 * 8, n1 * 4, n1 * 2, n1]

        self.Maxpool1_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Maxpool1_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Maxpool1_3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Maxpool1_4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.Conv1_1 = seq_conv_block(in_ch, filters[0])
        self.Conv1_2 = seq_conv_block(filters[0], filters[1])
        self.Conv1_3 = seq_conv_block(filters[1], filters[2])
        self.Conv1_4 = seq_conv_block(filters[2], filters[3])
        self.Conv1_5 = seq_conv_block(filters[3], filters[4])

        self.Up1_5 = seq_up_conv(filters[4], filters[3])
        self.Up_conv1_5 = seq_conv_block(filters[3] * 3, filters[3])

        self.Up1_4 = seq_up_conv(filters[3], filters[2])
        self.Up_conv1_4 = seq_conv_block(filters[2] * 3, filters[2])

        self.Up1_3 = seq_up_conv(filters[2], filters[1])
        self.Up_conv1_3 = seq_conv_block(filters[1] * 3, filters[1])

        self.Up1_2 = seq_up_conv(filters[1], filters[0])
        self.Up_conv1_2 = seq_conv_block(filters[0] * 3, filters[0])

        self.Conv1 = nn.Conv1d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.Maxpool2_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Maxpool2_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Maxpool2_3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Maxpool2_4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.Conv2_1 = seq_conv_block(in_ch, filters[0])
        self.Conv2_2 = seq_conv_block(filters[0], filters[1])
        self.Conv2_3 = seq_conv_block(filters[1], filters[2])
        self.Conv2_4 = seq_conv_block(filters[2], filters[3])
        self.Conv2_5 = seq_conv_block(filters[3], filters[4])

        self.Up2_5 = seq_up_conv(filters[4], filters[3])
        self.Up_conv2_5 = seq_conv_block(filters[3] * 2, filters[3])

        self.Up2_4 = seq_up_conv(filters[3], filters[2])
        self.Up_conv2_4 = seq_conv_block(filters[2] * 2, filters[2])

        self.Up2_3 = seq_up_conv(filters[2], filters[1])
        self.Up_conv2_3 = seq_conv_block(filters[1] * 2, filters[1])

        self.Up2_2 = seq_up_conv(filters[1], filters[0])
        self.Up_conv2_2 = seq_conv_block(filters[0] * 2, filters[0])

        self.Conv2 = nn.Conv1d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x2, x1):
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)

        e1_1 = self.Conv1_1(x1)

        e1_2 = self.Maxpool1_1(e1_1)
        e1_2 = self.Conv1_2(e1_2)

        e1_3 = self.Maxpool1_2(e1_2)
        e1_3 = self.Conv1_3(e1_3)

        e1_4 = self.Maxpool1_3(e1_3)
        e1_4 = self.Conv1_4(e1_4)

        e1_5 = self.Maxpool1_4(e1_4)
        e1_5 = self.Conv1_5(e1_5)

        e2_1 = self.Conv2_1(x2)

        e2_2 = self.Maxpool2_1(e2_1)
        e2_2 = self.Conv2_2(e2_2)

        e2_3 = self.Maxpool2_2(e2_2)
        e2_3 = self.Conv2_3(e2_3)

        e2_4 = self.Maxpool2_3(e2_3)
        e2_4 = self.Conv2_4(e2_4)

        e2_5 = self.Maxpool2_4(e2_4)
        e2_5 = self.Conv2_5(e2_5)
        d1_5 = self.Up1_5(e1_5)
        if d1_5.size(2) != e1_4.size(2):
            d1_5 = F.interpolate(d1_5, size=e1_4.size(2), mode='nearest')
        d1_5 = torch.cat((e1_4, e2_4, d1_5), dim=1)
        d1_5 = self.Up_conv1_5(d1_5)

        d1_4 = self.Up1_4(d1_5)
        if d1_4.size(2) != e1_3.size(2):
            d1_4 = F.interpolate(d1_4, size=e1_3.size(2), mode='nearest')
        d1_4 = torch.cat((e1_3, e2_3, d1_4), dim=1)
        d1_4 = self.Up_conv1_4(d1_4)

        d1_3 = self.Up1_3(d1_4)
        if d1_3.size(2) != e1_2.size(2):
            d1_3 = F.interpolate(d1_3, size=e1_2.size(2), mode='nearest')
        d1_3 = torch.cat((e1_2, e2_2, d1_3), dim=1)
        d1_3 = self.Up_conv1_3(d1_3)

        d1_2 = self.Up1_2(d1_3)
        if d1_2.size(2) != e1_1.size(2):
            d1_2 = F.interpolate(d1_2, size=e1_1.size(2), mode='nearest')
        d1_2 = torch.cat((e1_1, e2_1, d1_2), dim=1)
        d1_2 = self.Up_conv1_2(d1_2)

        out1 = self.Conv1(d1_2)
		
        d2_5 = self.Up2_5(e2_5)
        if d2_5.size(2) != e2_4.size(2):
            d2_5 = F.interpolate(d2_5, size=e2_4.size(2), mode='nearest')
        d2_5 = torch.cat((e2_4, d2_5), dim=1)
        d2_5 = self.Up_conv2_5(d2_5)

        d2_4 = self.Up2_4(d2_5)
        if d2_4.size(2) != e2_3.size(2):
            d2_4 = F.interpolate(d2_4, size=e2_3.size(2), mode='nearest')
        d2_4 = torch.cat((e2_3, d2_4), dim=1)
        d2_4 = self.Up_conv2_4(d2_4)

        d2_3 = self.Up2_3(d2_4)
        if d2_3.size(2) != e2_2.size(2):
            d2_3 = F.interpolate(d2_3, size=e2_2.size(2), mode='nearest')
        d2_3 = torch.cat((e2_2, d2_3), dim=1)
        d2_3 = self.Up_conv2_3(d2_3)

        d2_2 = self.Up2_2(d2_3)
        if d2_2.size(2) != e2_1.size(2):
            d2_2 = F.interpolate(d2_2, size=e2_1.size(2), mode='nearest')
        d2_2 = torch.cat((e2_1, d2_2), dim=1)
        d2_2 = self.Up_conv2_2(d2_2)

        out2 = self.Conv2(d2_2)
        out1 = out1.permute(0, 2, 1)
        out2 = out2.permute(0, 2, 1)

        return out1, out2