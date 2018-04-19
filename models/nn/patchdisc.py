"""GAN base Module"""

import torch
import torch.nn as nn

from models.base import Norm, initialize_weights, conv3x3

class PatchDiscriminator(nn.Module):
    def __init__(self, input_dim, norm_layer, use_sigmoid, gpu_ids):
        super().__init__()
        self.norm_layer = norm_layer
        self.gpu_ids = gpu_ids
        self.activation = nn.LeakyReLU(0.2, inplace=False)
        self.spectral = False
        self.conv = nn.Conv2d

        self.net = nn.Sequential
        net = [self._conv(input_dim, 64, 4, 2),
            self.activation,
            self._conv(64, 128, 4, 2),
            self.activation,
            self._conv(128, 256, 4, 2),
            self.activation,
            self._conv(256, 512, 4, 2),
            self.activation,
            self.conv(512, 1, 4, 1, 2)]
        if use_sigmoid:
            net.append(nn.Sigmoid)
        self.net = nn.Sequential(*net)
        initialize_weights(self, nn.init.kaiming_normal_)

    def _conv(self, indim, outdim, ksize, stride):
        c = self.conv(indim, outdim, ksize, stride)
        if self.spectral:
            return c
        else:
            return nn.Sequential(c, self.norm_layer(outdim))

    def forward(self, x):
        if len(self.gpu_ids) and isinstance(x.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.net, x, self.gpu_ids)
        else:
            return self.net(x)


class MultiscaleDiscriminator(PatchDiscriminator):
    def __init__(self, input_dim, spectral=False):
        super().__init__(input_dim, spectral)
        self.fine = nn.Sequential(
            self.activation,
            self._conv(512, 512, 4, 2),
        )
        self.fine_classifier = nn.Sequential(
            self.activation,
            self.conv(512, 1, 4, 1, 2),
            nn.Sigmoid(),
        )
        initialize_weights(self, nn.init.kaiming_normal_)

    def forward(self, x):
        f = self.features(x)
        coarse = self.coarse(f)
        coarse_out = self.classifier(coarse)

        fine = self.fine(coarse)
        fine_out = self.fine_classifier(fine)
        features = torch.cat((f.view(-1), coarse.view(-1), fine.view(-1)))
        return fine_out, coarse_out, features

