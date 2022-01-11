import torch
from torch import nn
from torch.utils import model_zoo


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False) -> None:
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel//2)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.bias.data.zero_()
    
    def forward(self, input):
        output = self.conv(input)
        if self.use_bn:
            output = self.conv(output)
        if self.activation:
            output = self.activation(output)

        return output 

class BasePool(nn.Module):
    def __init__(self, windows=2) -> None:
        super(BasePool, self).__init__()
        self.bn = nn.MaxPool2d(windows)
    
    def forward(self, input):
        output = self.bn(input)
        return output

class UpSampling(nn.Module):
    def __init__(self, factor=2) -> None:
        super(UpSampling, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
    
    def forward(self, input):
        output = self.up(input)
        return output

class VGG(nn.Module):
    def __init__(self) -> None:
        super(VGG, self).__init__()

        self.block_1 = nn.Sequential(
            BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True), 
            BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True),
            BasePool(2),
        )
        self.block_2 = nn.Sequential(
            BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True),
        ) #output 2_2

        self.block_3 = nn.Sequential(
            BasePool(2),
            BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True),
        ) #output 3_3

        self.block_4 = nn.Sequential(
            BasePool(2),
            BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True),
        ) #output 4_3

        self.block_5 = nn.Sequential(
            BasePool(2),
            BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True),
        ) #output 5_3

    def forward(self, input):

        output = self.block_1(input)

        conv2_2 = self.block_2(output)

        conv3_3 = self.block_3(conv2_2)

        conv4_3 = self.block_4(conv3_3)

        conv5_3 = self.block_5(conv4_3)

        return conv2_2, conv3_3, conv4_3, conv5_3

class MapPath(nn.Module):
    def __init__(self) -> None:
        super(MapPath, self).__init__()
        pass 