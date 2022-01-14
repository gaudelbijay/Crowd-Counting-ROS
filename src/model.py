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
        self.upsample = UpSampling(factor=2)

        self.conv1 = BaseConv(1024, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv3 = BaseConv(512, 128, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv5 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, *input):
        conv2_2, conv3_3, conv4_3, conv5_3 = input

        input = self.upsample(conv5_3)

        input = torch.cat([input, conv4_3], 1)
        input = self.conv1(input)
        input = self.conv2(input)
        input = self.upsample(input)

        input = torch.cat([input, conv3_3], 1)
        input = self.conv3(input)
        input = self.conv4(input)
        input = self.upsample(input)

        input = torch.cat([input, conv2_2], 1)
        input = self.conv5(input)
        input = self.conv6(input)
        input = self.conv7(input)

        return input


class ModelNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vgg = VGG()
        self.amp = MapPath()
        self.dmp = MapPath()

        self.conv_att = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=True)
        self.conv_out = BaseConv(32, 1, 1, 1, activation=None, use_bn=False)
    
    def load_vgg(self):
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
        old_name = [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18, 20, 21, 24, 25, 27, 28, 30, 31, 34, 35, 37, 38, 40, 41]
        new_name = [1.0, 1.1, 2.0, 2.1, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 5.1, 5.2, 5.3]
        new_dict = {}
        print(state_dict.keys())
        for i in range(len(new_name)):
            new_dict['block' + str(new_name[i]) + '.conv.weight'] = state_dict['features.'+str(old_name[2*i])+'.weight']
            new_dict['block' + str(new_name[i]) + '.conv.bias'] = state_dict['features.' + str(old_name[2 * i]) + '.bias']
            new_dict['block' + str(new_name[i]) + '.bn.weight'] = state_dict['features.' + str(old_name[2 * i + 1]) + '.weight']
            new_dict['block' + str(new_name[i]) + '.bn.bias'] = state_dict['features.' + str(old_name[2 * i + 1]) + '.bias']
            new_dict['block' + str(new_name[i]) + '.bn.running_mean'] = state_dict['features.' + str(old_name[2 * i + 1]) + '.running_mean']
            new_dict['block' + str(new_name[i]) + '.bn.running_var'] = state_dict['features.' + str(old_name[2 * i + 1]) + '.running_var']

        self.vgg.load_state_dict(new_dict)

    def forward(self, input):
        input = self.vgg(input)
        amp_out = self.amp(*input)
        dmp_out = self.dmp(*input)

        amp_out = self.conv_att(amp_out)
        dmp_out = amp_out * dmp_out
        dmp_out = self.conv_out(dmp_out)

        return dmp_out, amp_out


if __name__ == '__main__':
    input = torch.randn(8, 3, 400, 400)
    model = ModelNetwork()
    output, attention = model(input)
    print(input.size())
    print(output.size())
    print(attention.size())