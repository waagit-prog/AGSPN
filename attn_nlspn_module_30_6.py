import torch
import torch.nn as nn
# from .modulated_deform_conv_func import ModulatedDeformConvFunction
from src.config import args as args_config
from src.model.modulated_deform_conv_func import ModulatedDeformConvFunction
import math
import numpy as np
from torch.nn.modules.utils import _pair as to_2tuple
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from functools import partial
from timm.models.layers import DropPath
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(stride, stride),
                     padding=(1, 1), bias=False)


def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True,
                 relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding,
                            bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers


class Block(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.stride = stride
        self.fc1 = conv_bn_relu(inplanes, planes, kernel=3, padding=1, bn=False)
        self.fc2 = conv_bn_relu(inplanes, planes, kernel=3, padding=1, bn=False)

    def forward(self, x):
        # x = self.fc1(x)
        # x = self.fc2(x)
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return out


class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_0_ = nn.Conv2d(dim, dim, 1, padding=0, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, dilation=1)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze_ = nn.Conv2d(2, 2, 3, padding=1)
        self.conv = nn.Conv2d(dim // 2, 3, 1)

    def forward(self, y, x):
        attn1 = self.conv_0_(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze_(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return torch.sum(y * attn, dim=1, keepdim=True)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class Affinity(nn.Module):
    def __init__(self, ch_g, propagation_kernel=3, aff_kernel=1):
        super(Affinity, self).__init__()
        self.ch_g = ch_g
        self.conf_prop = False
        self.k_g = propagation_kernel
        pad_g = int((aff_kernel - 1) / 2)

        self.num = propagation_kernel * propagation_kernel - 1
        self.idx_ref = self.num // 2

        self.aff_weight = nn.Conv2d(
            self.ch_g, self.k_g ** 2, kernel_size=aff_kernel, stride=1,
            padding=pad_g, bias=True
        )
        self.aff_offset = nn.Conv2d(
            self.ch_g, 2 * (self.k_g ** 2 - 1), kernel_size=aff_kernel, stride=1,
            padding=pad_g, bias=True
        )

        self.aff_weight.weight.data.zero_()
        self.aff_weight.bias.data.zero_()

        self.aff_offset.weight.data.zero_()
        self.aff_offset.bias.data.zero_()

    def forward(self, feature, conf):
        B, _, H, W = feature.shape

        # feature = self.conv2(self.conv1(feature))

        weight = torch.sigmoid(self.aff_weight(feature))

        if self.conf_prop:
            aff_list = list(torch.chunk(weight, self.k_g ** 2, dim=1))
            rec = aff_list[self.k_g ** 2 // 2]

            weight_temp = weight * conf
            weight_temp_list = list(torch.chunk(weight_temp, self.k_g ** 2, dim=1))
            weight_temp_list[self.k_g ** 2 // 2] = rec
            weight = torch.cat(weight_temp_list, dim=1).view(B, self.k_g ** 2, H, W)

        weight = weight / (torch.sum(weight, 1).unsqueeze(1).expand_as(weight) + 1e-8)

        # Add zero reference offset
        offset = self.aff_offset(feature)
        offset = offset.view(B, self.num, 2, H, W)
        list_offset = list(torch.chunk(offset, self.num, dim=1))
        list_offset.insert(self.idx_ref,
                           torch.zeros((B, 1, 2, H, W)).type_as(offset))
        offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)

        return offset, weight


class NLSPN(nn.Module):
    def __init__(self, args, ch_g, ch_f, k_g, k_f):
        super(NLSPN, self).__init__()

        # Guidance : [B x ch_g x H x W]
        # Feature : [B x ch_f x H x W]

        # Currently only support ch_f == 1
        assert ch_f == 1, 'only tested with ch_f == 1 but {}'.format(ch_f)

        assert (k_g % 2) == 1, \
            'only odd kernel is supported but k_g = {}'.format(k_g)

        pad_g_3 = int((3 - 1) / 2)
        assert (k_f % 2) == 1, \
            'only odd kernel is supported but k_f = {}'.format(k_f)

        pad_f_3 = int((3 - 1) / 2)

        self.args = args
        self.prop_time = self.args.prop_time
        self.affinity = self.args.affinity

        self.ch_g = ch_g
        self.ch_f = ch_f
        self.k_g = k_g

        self.k_g_3 = 3

        self.k_f_3 = 3
        # Assume zero offset for center pixels
        # self.num_3 = 3 * 3 - 1
        # self.idx_ref_3 = self.num_3 // 2

        self.atten = LSKblock(16 + 6)

        self.proj = conv_bn_relu(3, 6, kernel=1, stride=1)

        # Dummy parameters for gathering
        self.w_3_ = nn.Parameter(torch.ones((self.ch_f, 1, self.k_f_3, self.k_f_3)))
        self.b_3_ = nn.Parameter(torch.zeros(self.ch_f))

        self.w_3_.requires_grad = False
        self.b_3_.requires_grad = False

        self.stride = 1
        self.padding_3 = pad_f_3
        self.dilation = 1
        self.groups = self.ch_f
        self.deformable_groups = 1
        self.im2col_step = 64

        self._get_offset_affinity_3_1_ = Affinity(8, 3, 3)
        self._get_offset_affinity_3_2_ = Affinity(8, 3, 3)
        self._get_offset_affinity_3_3_ = Affinity(8, 3, 3)
        self._get_offset_affinity_3_4_ = Affinity(8, 3, 3)
        self._get_offset_affinity_3_5_ = Affinity(8, 3, 3)
        self._get_offset_affinity_3_6_ = Affinity(8, 3, 3)

    def _propagate_once(self, feat, offset, aff):
        feat = ModulatedDeformConvFunction.apply(
            feat, offset, aff, self.w_3_, self.b_3_, self.stride, self.padding_3,
            self.dilation, self.groups, self.deformable_groups, self.im2col_step
        )
        return feat

    def combine(self, feat, attn):

        self_feat = self.proj(feat)
        attn = torch.cat([attn, self_feat], dim=1)
        dep = self.atten(feat, attn)
        return dep

    def forward(self, feat_init, guidance, conf=None, attn=None, feat_fix=None,
                rgb=None):
        guidance_list = torch.chunk(guidance, self.prop_time, dim=1)

        # assert self.ch_g == guidance_list[0].shape[1]
        # assert self.ch_f == feat_init.shape[1]

        offset_1, aff_1 = self._get_offset_affinity_3_1_(guidance_list[0], conf)

        offset_2, aff_2 = self._get_offset_affinity_3_2_(guidance_list[1], conf)

        offset_3, aff_3 = self._get_offset_affinity_3_3_(guidance_list[2], conf)

        offset_4, aff_4 = self._get_offset_affinity_3_4_(guidance_list[3], conf)

        offset_5, aff_5 = self._get_offset_affinity_3_5_(guidance_list[4], conf)

        offset_6, aff_6 = self._get_offset_affinity_3_6_(guidance_list[5], conf)

        mask_fix = feat_fix.sign()
        feat_result = feat_init.float().contiguous()
        list_feat = []

        offset_list = [offset_1, offset_2, offset_3, offset_4, offset_5, offset_6]
        aff_list = [aff_1, aff_2, aff_3, aff_4, aff_5, aff_6]

        for k in range(0, self.prop_time):
            # Input preservation for each iteration
            if False:
                feat_result = (1.0 - mask_fix) * feat_result \
                              + mask_fix * feat_fix

            feat_result = self._propagate_once(feat_result, offset_list[k], aff_list[k])

            list_feat.append(feat_result)

        feat_result = self.combine(torch.cat([list_feat[3], list_feat[4], list_feat[5]], dim=1), attn)
        list_feat.append(feat_result)

        return feat_result, list_feat, offset_1, offset_2, offset_3, offset_4, offset_5, offset_6, aff_1, self.w_3_.data


if __name__ == '__main__':
    a = torch.randn((1, 1, 48, 48)).to('cuda')
    guide = torch.ones((1, 6 * 16, 48, 48)).to('cuda')
    conf = torch.randn((1, 1, 48, 48)).to('cuda')

    attn = torch.randn((1, 16, 48, 48)).to('cuda')

    d = NLSPN(args_config, ch_g=8, ch_f=1, k_g=3, k_f=3).to('cuda')
    print(d(a, guide, conf, attn, a)[0].shape)