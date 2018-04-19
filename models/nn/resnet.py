"""ResNet base Module"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import Norm, conv1x1, conv3x3, initialize_weights, selu_init, Bottlenext, Bottleneck


class ResNet(nn.Module):
    """ResNet Baseclass which defines the _make_layer function
    """
    inplanes = 64

    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def _make_layer(self, block, planes, repetitions, stride=1, dilation=1, reverse=False, norm=True, dropout=None):
        """Assembles the given block to a concatenation of blocks"""
        downsample = None
        if reverse:
            indim = self.inplanes
            outdim = self.inplanes // block.expansion
        else:
            indim = planes * block.expansion
            outdim = planes

        layers = []
        for _ in range(1, repetitions):
            layers.append(block(indim, outdim, dilation=dilation))
            if dropout is not None:
                layers.append(nn.Dropout2d(dropout))

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds = [conv1x1(self.inplanes, planes * block.expansion, stride=stride, spectral=block.spectral)]
            # if stride != 1:
            #     ds.append(nn.AvgPool2d(2))
            if not block.spectral:
                ds.append(Norm(planes * block.expansion))
            downsample = nn.Sequential(*ds)

        adjustment_layer = block(
            self.inplanes, planes, stride, dilation, downsample)
        self.inplanes = planes * block.expansion

        if reverse:
            layers.append(adjustment_layer)
        else:
            layers.insert(0, adjustment_layer)

        return nn.Sequential(*layers)

    def from_checkpoint(self, path):
        self.load_state_dict(torch.load(path)['state_dict'], strict=False)


    def forward(self, *inputs):
        raise NotImplementedError
