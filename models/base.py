"""Defenition of the various architectures"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Norm = nn.BatchNorm2d
Norm = nn.InstanceNorm2d


def initialize_weights(model, init):
    """Initialize the given model according to a chosen
       function
    """
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            init(layer.weight.data)
        elif isinstance(layer, nn.ConvTranspose2d):
            init(layer.weight.data)
        elif isinstance(layer, nn.BatchNorm2d):
            layer.weight.data.normal_(1.0, 0.2)
            layer.bias.data.zero_()


def selu_init(tensor):
    import torch.nn.init as init
    import math
    fan = init._calculate_correct_fan(tensor, 'fan_in')
    std = math.sqrt(1 / fan)
    return tensor.normal_(0, std)


def conv1x1(in_planes, out_planes, stride=1, padding=0, spectral=False):
    """1x1 Convolution Helper function"""
    if spectral:
        conv = nn.Conv2d
    else:
        conv = nn.Conv2d
    return conv(in_planes, out_planes, kernel_size=1,
                     stride=stride, padding=padding, bias=False)

def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=None, groups=1, spectral=False):
    "3x3 Convolution Helper function"""
    if dilation is not None:
        padding = dilation
    else:
        dilation = 1
    if spectral:
        conv = nn.Conv2d
    else:
        conv = nn.Conv2d

    return conv(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, groups=groups ,bias=False)


class Bottlenext(nn.Module):
    """Aggregated pre activation Bottleneck block
       arXiv:1611.05431v2
    """
    expansion = 4
    dim = 1
    cardinality = 32
    def __init__(self, inplanes, planes, stride=1, dilation=None, downsample=None, spectral=False):
        super(Bottlenext, self).__init__()
        assert planes % self.cardinality == 0
        expanse = planes * self.dim
        # actif = nn.SELU()
        self.act = nn.ReLU(inplace=True)

        self.norm1 = Norm(inplanes)
        self.conv1 = conv1x1(inplanes, expanse)

        self.norm2 = Norm(expanse)
        self.conv2 = conv3x3(expanse, expanse, stride=stride, dilation=dilation, groups=self.cardinality)

        self.norm3 = Norm(expanse)
        self.conv3 = conv1x1(expanse, planes * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Forward Pass"""
        residual = x

        out = self.norm1(x)
        out = self.act(out)
        out = self.conv1(out)


        out = self.norm2(out)
        out = self.act(out)
        out = self.conv2(out)

        out = self.norm3(out)
        out = self.act(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, spectral=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = Norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = Norm(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = Norm(planes * self.expansion)
        self.act = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1
    activation = nn.ReLU(inplace=True)
    spectral = False

    def __init__(self, inplanes, planes, stride=1, dilation=None, downsample=None):
        super().__init__()


        self.conv1 = conv3x3(inplanes, planes, stride, spectral=self.spectral)
        self.conv2 = conv3x3(planes, planes, spectral=self.spectral)

        if self.spectral:
            self.bn1 = None 
            self.bn2 = None 
        else:
            self.bn1 = Norm(planes)
            self.bn2 = Norm(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out = out + residual
        out = self.activation(out)

        return out




class Upsample(nn.Module):
    """Applies Upsampling and 1x1 convolution + activation
       operations to a given tensor
    """

    def __init__(self, in_planes, out_planes, ks, activation):
        super().__init__()
        self.up_pool = nn.Upsample(scale_factor=2, mode='bilinear')
        if ks == 3:
            self.plane_conv = conv3x3(in_planes, out_planes)
        else:
            self.plane_conv = conv1x1(in_planes, out_planes)
        self.norm = Norm(out_planes)
        self.activation = activation

    def forward(self, x):
        """Forward Pass"""
        x = self.up_pool(x)
        x = self.plane_conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class SNBlock(nn.Module):
    expansion = 1
    activation = nn.ReLU(inplace=True)
    spectral = True

    def __init__(self, inplanes, planes, downsample=None):
        super().__init__()

        self.conv1 = conv3x3(inplanes, planes, spectral=self.spectral)
        self.conv2 = conv3x3(planes, planes, spectral=self.spectral)
        self.avg_pool = nn.AvgPool2d(2)


        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.activation(out)

        out = self.avg_pool(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out = out + residual

        return out
