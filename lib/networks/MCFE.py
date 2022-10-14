from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import scipy.io as io

import math
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
BN_MOMENTUM = 0.1


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1   = nn.Conv3d(in_planes, in_planes // ratio, kernel_size=3, padding=1, bias=False)
        self.relu = nn.LeakyReLU(inplace=True)
        self.fc2   = nn.Conv3d(in_planes // ratio, in_planes, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes * self.expansion)
        self.bn2 = nn.BatchNorm3d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=True,
                 widen_factor=1.0):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.channels = block_inplanes

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 1, 1),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)

        self.bn1 = nn.BatchNorm3d(self.in_planes)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(inplace=True)

        self.ca1 = ChannelAttention(self.in_planes)
        self.sa1 = SpatialAttention()

        self.layer0 = self._make_layer(block, block_inplanes[0],
                                       layers[0],
                                       stride=2)
        self.layer1 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       stride=2)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[4],
                                       layers[4],
                                       stride=2)

        self.ca2 = ChannelAttention(self.in_planes)
        self.sa2 = SpatialAttention()


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.in_planes, planes * block.expansion, (1, stride, stride)),
                nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=(1, stride, stride),
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        y = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.ca1(x) * x
        x = self.sa1(x) * x

        y.append(x)
        for i in range(5):
            x = getattr(self, 'layer{}'.format(i))(x)
            if i == 4:
                x = self.ca2(x) * x
                x = self.sa2(x) * x
            y.append(x)

        return y



def resnet3D():
    model = ResNet3D(BasicBlock, [3, 3, 3, 3, 3], [16, 32, 64, 128, 256, 512])
    return model


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]




class fuse_decode(nn.Module):
    def __init__(self, o, channels, up_f):
        super(fuse_decode, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            fuse = nn.Sequential(conv3x3x3(o, o),
                nn.BatchNorm3d(o))

            up = nn.ConvTranspose3d(c, o,
                                    (3, f*2, f * 2),
                                    stride=(1, f, f),
                                    padding=(1, f//2, f//2),
                                    output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'up_' + str(i), up)
            setattr(self, 'fuse_' + str(i), fuse)


    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            layers[i] = upsample(layers[i])
            fuse = getattr(self, 'fuse_' + str(i - startp))
            layers[i] = fuse(layers[i] + layers[i - 1])


class Upsample(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(Upsample, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'fd_{}'.format(i),
                    fuse_decode(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]
        for i in range(len(layers) - self.startp - 1):
            fd = getattr(self, 'fd_{}'.format(i))
            fd(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x


class MultiCNN(nn.Module):
    def __init__(self, pretrained, down_ratio, last_level, out_channel=0):
        super(MultiCNN, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = resnet3D()
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.up = Upsample(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.fd = fuse_decode(out_channel, channels[self.first_level:self.last_level],
                                [2 ** i for i in range(self.last_level - self.first_level)])

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.base(x)
        x = self.up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.fd(y, 0, len(y))

        return y[-1]