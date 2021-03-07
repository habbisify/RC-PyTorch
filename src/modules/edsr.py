"""
Copyright 2020, ETH Zurich

This file is part of RC-PyTorch.

RC-PyTorch is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

RC-PyTorch is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RC-PyTorch.  If not, see <https://www.gnu.org/licenses/>.
"""
# Code adapted from src/model/common.py of this repo:
#
# https://github.com/thstkdgus35/EDSR-PyTorch
#
# Original license:
#
# MIT License
#
# Copyright (c) 2018 Sanghyun Son
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math

import torch
import torch.nn as nn

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

# previous residual block
class ResBlock_old(nn.Module):
    def __init__(self, conv, n_feats, kernel_size,
                 bias=True, bn=False, norm_cls=None, act=nn.ReLU(True), res_scale=1):

        super(ResBlock_old, self).__init__()
        m = []
        _repr = []
        # do this two times (CONV, GDN, ReLU)
        for i in range(2):
            # CONV append 
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            _repr.append(f'Conv({n_feats}x{kernel_size})')
            # Normalization append (either BN or GDN)
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
                _repr.append(f'BN({n_feats})')
            elif norm_cls is not None:
                m.append(norm_cls())
                _repr.append(f'N({n_feats})')
            # ReLU append (only for the first time)
            if i == 0:
                m.append(act)
                _repr.append(repr(act))
        if res_scale != 1:
            _repr.append(f'res_scale={res_scale}')
        self.res_scale = res_scale
        self.body = nn.Sequential(*m)
        self._repr = '/'.join(_repr)

    def forward_old(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

    def __repr__(self):
        return f'ResBlock_old({self._repr})'

# new "Bottleneck" residual block (gdn_wide_deep3.cf: conv=pytorch_default_conv, n_feats=Cf = 128, kernel_size = 3)
class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size,
                 bias=True, bn=False, norm_cls=None, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        _repr = []
        # blocks are 
        # (1): 1x1, 128 (1x1, 64 in ResNet paper)
        # (2): 3x3, 128 (3x3, 64 in ResNet paper)
        # (3): 1x1, 512 (1x1, 256 in ResNet paper)
        n_feats = [128, 128, 128]
        kernel_size = [1, 3, 1]
        for i in range(3):
            m.append(conv(n_feats[i], n_feats[i], kernel_size[i], bias=bias))
            _repr.append(f'Conv({n_feats[i]}x{kernel_size[i]})')
            # Normalization remains unaffected
            if bn:
                m.append(nn.BatchNorm2d(n_feats[i]))
                _repr.append(f'BN({n_feats[i]})')
            elif norm_cls is not None:
                m.append(norm_cls())
                _repr.append(f'N({n_feats[i]})')
            # ReLU only after first and second conv
            if i < 2:
                m.append(act)
                _repr.append(repr(act))
        if res_scale != 1:
            _repr.append(f'res_scale={res_scale}') 
        self.res_scale = res_scale
        self.body = nn.Sequential(*m)
        self._repr = '/'.join(_repr)

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

    def __repr__(self):
        return f'ResBlock({self._repr})'

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

