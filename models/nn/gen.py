from collections import OrderedDict

import torch
import torch.nn as nn

from models.base import BasicBlock, conv3x3, Norm, initialize_weights
from models.nn.resnet import ResNet


class GlobalGenerator(ResNet):
    block = BasicBlock
    def __init__(self, input_dim, output_dim, norm_layer, dropout, gpu_ids):
        super().__init__(layers=[9])
        self.norm_layer = norm_layer
        self.gpu_ids = gpu_ids
        self.activation = nn.ReLU(inplace=True)
        self.inplanes = 1024

        dropout = 0.5 if dropout else None

        encoder = nn.Sequential(OrderedDict([
            ('c7s1-64', nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(input_dim, 64, 7),
                self.norm_layer(64),
                self.activation)),
            ('d128', nn.Sequential(
                conv3x3(64, 128, stride=2),
                self.norm_layer(128),
                self.activation)),
            ('d256', nn.Sequential(
                conv3x3(128, 256, stride=2),
                self.norm_layer(256),
                self.activation)),
            ('d512', nn.Sequential(
                conv3x3(256, 512, stride=2),
                self.norm_layer(512),
                self.activation)),
            ('d1024', nn.Sequential(
                conv3x3(512, 1024, stride=2),
                self.norm_layer(1024),
                self.activation)),
            ('R1024', self._make_layer(
                self.block, 1024, self.layers[0], dropout=dropout)),
        ]))

        decoder_layers = OrderedDict([
            ('u512', nn.Sequential(
                nn.ConvTranspose2d(
                    1024, 512, 3, 2, padding=1, output_padding=1),
                self.norm_layer(512),
                self.activation)),
            ('u256', nn.Sequential(
                nn.ConvTranspose2d(
                    512, 256, 3, 2, padding=1, output_padding=1),
                self.norm_layer(256),
                self.activation)),
            ('u128', nn.Sequential(
                nn.ConvTranspose2d(
                    256, 128, 3, 2, padding=1, output_padding=1),
                self.norm_layer(128),
                self.activation)),
            ('u64', nn.Sequential(
                nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1),
                self.norm_layer(64),
                self.activation)),
        ])
        if isinstance(output_dim, int):
            decoder_layers['c7s1-N'] = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(64, output_dim, 7),
                nn.Tanh())

        decoder = nn.Sequential(decoder_layers)
        self.global_gen = nn.Sequential(OrderedDict([('encoder', encoder), ('decoder', decoder)]))
        initialize_weights(self, nn.init.kaiming_normal_)

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            if isinstance(module, nn.Dropout2d):
                module.train(True)
            else:
                module.train(mode)
        return self

    def eval(self):
        self.train(mode=False)

    def forward(self, x):
        if self.gpu_ids and isinstance(x.data, torch.cuda.FloatTensor):
            x = nn.parallel.data_parallel(self.global_gen, x, self.gpu_ids)
        else:
            x = self.global_gen(x)
        return x


class LocalEnhancer(GlobalGenerator):
    # c7s1-32,d64,R64,R64,R64,u32,c7s1-3
    def __init__(self, input_dim, output_dim, norm_layer, use_dropout, gpu_ids, downsample=False):
        super().__init__(input_dim, None, norm_layer, use_dropout, gpu_ids)
        self.inplanes = 64
        self.downsample = downsample

        stride = 2 if self.downsample else 1

        self.local_ds = nn.Sequential(OrderedDict([
            ('c7s1-32', nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(input_dim, 32, 7),
                self.norm_layer(32),
                self.activation)),
            ('d64', nn.Sequential(
                conv3x3(32, 64, stride=stride),
                self.norm_layer(64),
                self.activation)),
        ]))

        conv = nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1) if self.downsample else conv3x3(64, 32)
        self.local_us = nn.Sequential(OrderedDict([
            ('R64', self._make_layer(self.block, 64, 3)),
            ('u32', nn.Sequential(
                conv,
                self.norm_layer(32),
                self.activation)),
            ('c7s1-3', nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(32, output_dim, 7),
                nn.Tanh()))
        ]))
        initialize_weights(self, nn.init.kaiming_normal_)

    def forward(self, x):
        if self.gpu_ids and isinstance(x.data, torch.cuda.FloatTensor):
            _down_sampled = nn.parallel.data_parallel(self.local_ds, x, self.gpu_ids)
            ds = self.down(x)
            _global = nn.parallel.data_parallel(self.global_gen, x, self.gpu_ids)
            out = nn.parallel.data_parallel(self.local_us, _global + _down_sampled, self.gpu_ids)
        else:
            _down_sampled = self.local_ds(x)
            _global = self.decoder(self.encoder(self.down(x)))
            out = self.local_us(_global + _down_sampled)
        return x

    def down(self, x):
        if self.downsample:
            return nn.functional.avg_pool2d(x, 3, stride=2, padding=[1, 1], count_include_pad=False)
        else:
            return x
