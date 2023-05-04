#modified from Github repo: https://github.com/JizhiziLi/P3M
#added inference code for other networks


import torch
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage.transform import resize
from torchvision import transforms,models
import os 
from models import *
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.nn.functional as fnn
import glob
import tqdm
from torch.autograd import Variable
from typing import Type, Any, Callable, Union, List, Optional
import logging 
import time
from omegaconf import OmegaConf
config = OmegaConf.load(os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "config/base.yaml"))
device = "cuda"

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class TFI(nn.Module):
    expansion = 1
    def __init__(self, planes,stride=1):
        super(TFI, self).__init__()
        middle_planes = int(planes/2)
        self.transform = conv1x1(planes, middle_planes)
        self.conv1 = conv3x3(middle_planes*3, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
    def forward(self, input_s_guidance, input_m_decoder, input_m_encoder):
        input_s_guidance_transform = self.transform(input_s_guidance)
        input_m_decoder_transform = self.transform(input_m_decoder)
        input_m_encoder_transform = self.transform(input_m_encoder)
        x = torch.cat((input_s_guidance_transform,input_m_decoder_transform,input_m_encoder_transform),1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out
class SBFI(nn.Module):
    def __init__(self, planes,stride=1):
        super(SBFI, self).__init__()
        self.stride = stride
        self.transform1 = conv1x1(planes, int(planes/2))
        self.transform2 = conv1x1(64, int(planes/2))
        self.maxpool = nn.MaxPool2d(2, stride=stride)
        self.conv1 = conv3x3(planes, planes, 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, input_m_decoder,e0):
        input_m_decoder_transform = self.transform1(input_m_decoder)
        e0_maxpool = self.maxpool(e0)
        e0_transform = self.transform2(e0_maxpool)
        x = torch.cat((input_m_decoder_transform,e0_transform),1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = out+input_m_decoder
        return out
class DBFI(nn.Module):
    def __init__(self, planes,stride=1):
        super(DBFI, self).__init__()
        self.stride = stride
        self.transform1 = conv1x1(planes, int(planes/2))
        self.transform2 = conv1x1(512, int(planes/2))
        self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
        self.conv1 = conv3x3(planes, planes, 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, 3, 1)
        self.upsample2 = nn.Upsample(scale_factor=int(32/stride), mode='bilinear')
    def forward(self, input_s_decoder,e4):
        input_s_decoder_transform = self.transform1(input_s_decoder)
        e4_transform = self.transform2(e4)
        e4_upsample = self.upsample(e4_transform)
        x = torch.cat((input_s_decoder_transform,e4_upsample),1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = out+input_s_decoder
        out_side = self.conv2(out)
        out_side = self.upsample2(out_side)
        return out, out_side
class P3mNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet34_mp()
        ############################
        ### Encoder part - RESNETMP
        ############################
        self.encoder0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            )
        self.mp0 = self.resnet.maxpool1
        self.encoder1 = nn.Sequential(
            self.resnet.layer1)
        self.mp1 = self.resnet.maxpool2
        self.encoder2 = self.resnet.layer2
        self.mp2 = self.resnet.maxpool3
        self.encoder3 = self.resnet.layer3
        self.mp3 = self.resnet.maxpool4
        self.encoder4 = self.resnet.layer4
        self.mp4 = self.resnet.maxpool5

        self.tfi_3 = TFI(256)
        self.tfi_2 = TFI(128)
        self.tfi_1 = TFI(64)
        self.tfi_0 = TFI(64)

        self.sbfi_2 = SBFI(128, 8)
        self.sbfi_1 = SBFI(64, 4)
        self.sbfi_0 = SBFI(64, 2)

        self.dbfi_2 = DBFI(128, 4)
        self.dbfi_1 = DBFI(64, 8)
        self.dbfi_0 = DBFI(64, 16)

        ##########################
        ### Decoder part - GLOBAL
        ##########################
        self.decoder4_g = nn.Sequential(
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear') )
        self.decoder3_g = nn.Sequential(
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear') )
        self.decoder2_g = nn.Sequential(
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        self.decoder1_g = nn.Sequential(
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        self.decoder0_g = nn.Sequential(
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,3,3,padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'))

        ##########################
        ### Decoder part - LOCAL
        ##########################
        self.decoder4_l = nn.Sequential(
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.decoder3_l = nn.Sequential(
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.decoder2_l = nn.Sequential(
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.decoder1_l = nn.Sequential(
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.decoder0_l = nn.Sequential(
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.decoder_final_l = nn.Conv2d(64,1,3,padding=1)

        
    def forward(self, input):
        ##########################
        ### Encoder part - RESNET
        ##########################
        e0 = self.encoder0(input)
        e0p, id0 = self.mp0(e0)
        e1p, id1 = self.mp1(e0p)
        e1 = self.encoder1(e1p)
        e2p, id2 = self.mp2(e1)
        e2 = self.encoder2(e2p)
        e3p, id3 = self.mp3(e2)
        e3 = self.encoder3(e3p)
        e4p, id4 = self.mp4(e3)
        e4 = self.encoder4(e4p)
        ###########################
        ### Decoder part - Global
        ###########################
        d4_g = self.decoder4_g(e4)
        d3_g = self.decoder3_g(d4_g)
        d2_g, global_sigmoid_side2 = self.dbfi_2(d3_g, e4)
        d2_g = self.decoder2_g(d2_g)
        d1_g, global_sigmoid_side1 = self.dbfi_1(d2_g, e4)
        d1_g = self.decoder1_g(d1_g)
        d0_g, global_sigmoid_side0 = self.dbfi_0(d1_g, e4)
        d0_g = self.decoder0_g(d0_g)
        global_sigmoid = d0_g
        ###########################
        ### Decoder part - Local
        ###########################
        d4_l = self.decoder4_l(e4)
        d4_l = F.max_unpool2d(d4_l, id4, kernel_size=2, stride=2)
        d3_l = self.tfi_3(d4_g, d4_l, e3)
        d3_l = self.decoder3_l(d3_l)
        d3_l = F.max_unpool2d(d3_l, id3, kernel_size=2, stride=2)
        d2_l = self.tfi_2(d3_g, d3_l, e2)
        d2_l = self.sbfi_2(d2_l, e0)
        d2_l = self.decoder2_l(d2_l)
        d2_l  = F.max_unpool2d(d2_l, id2, kernel_size=2, stride=2)
        d1_l = self.tfi_1(d2_g, d2_l, e1)
        d1_l = self.sbfi_1(d1_l, e0)
        d1_l = self.decoder1_l(d1_l)
        d1_l  = F.max_unpool2d(d1_l, id1, kernel_size=2, stride=2)
        d0_l = self.tfi_0(d1_g, d1_l, e0p)
        d0_l = self.sbfi_0(d0_l, e0)
        d0_l = self.decoder0_l(d0_l)
        d0_l  = F.max_unpool2d(d0_l, id0, kernel_size=2, stride=2)
        d0_l = self.decoder_final_l(d0_l)
        local_sigmoid = F.sigmoid(d0_l)
        ##########################
        ### Fusion net - G/L
        ##########################
        fusion_sigmoid = get_masked_local_from_global(global_sigmoid, local_sigmoid)
        return global_sigmoid, local_sigmoid, fusion_sigmoid, global_sigmoid_side2, global_sigmoid_side1, global_sigmoid_side0
    


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes,stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.attention(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        #pdb.set_trace()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1, idx1 = self.maxpool1(x1)

        x2, idx2 = self.maxpool2(x1)
        x2 = self.layer1(x2)

        x3, idx3 = self.maxpool3(x2)
        x3 = self.layer2(x3)

        x4, idx4 = self.maxpool4(x3)
        x4 = self.layer3(x4)

        x5, idx5 = self.maxpool5(x4)
        x5 = self.layer4(x5)
        
        x_cls = self.avgpool(x5)
        x_cls = torch.flatten(x_cls, 1)
        x_cls = self.fc(x_cls)

        return x_cls

    def forward(self, x):
        return self._forward_impl(x)


def resnet34_mp(**kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    checkpoint = torch.load("checkpoints/r34mp_pretrained_imagenet.pth.tar")
    model.load_state_dict(checkpoint)
    return model
        
##############################
### Training loses for P3M-NET
##############################
def get_crossentropy_loss(gt,pre):  
    gt_copy = gt.clone()
    gt_copy[gt_copy==0] = 0
    gt_copy[gt_copy==255] = 2
    gt_copy[gt_copy>2] = 1
    gt_copy = gt_copy.long()
    gt_copy = gt_copy[:,0,:,:]
    criterion = nn.CrossEntropyLoss()
    entropy_loss = criterion(pre, gt_copy)
    return entropy_loss

def get_alpha_loss(predict, alpha, trimap):
    weighted = torch.zeros(trimap.shape).cuda()
    weighted[trimap == 128] = 1.
    alpha_f = alpha / 255.
    alpha_f = alpha_f.cuda()
    diff = predict - alpha_f
    diff = diff * weighted
    alpha_loss = torch.sqrt(diff ** 2 + 1e-12)
    alpha_loss_weighted = alpha_loss.sum() / (weighted.sum() + 1.)
    return alpha_loss_weighted

def get_alpha_loss_whole_img(predict, alpha):
    weighted = torch.ones(alpha.shape).cuda()
    alpha_f = alpha / 255.
    alpha_f = alpha_f.cuda()
    diff = predict - alpha_f
    alpha_loss = torch.sqrt(diff ** 2 + 1e-12)
    alpha_loss = alpha_loss.sum()/(weighted.sum())
    return alpha_loss

## Laplacian loss is refer to 
## https://gist.github.com/MarcoForte/a07c40a2b721739bb5c5987671aa5270
def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size,0:size].T)
    gaussian = lambda x: np.exp((x - size//2)**2/(-2*sigma**2))**2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    kernel = np.tile(kernel, (n_channels, 1, 1))
    kernel = torch.FloatTensor(kernel[:, None, :, :]).cuda()
    return Variable(kernel, requires_grad=False)

def conv_gauss(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    img = fnn.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
    return fnn.conv2d(img, kernel, groups=n_channels)

def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = fnn.avg_pool2d(filtered, 2)
    pyr.append(current)
    return pyr

def get_laplacian_loss(predict, alpha, trimap):
    weighted = torch.zeros(trimap.shape).cuda()
    weighted[trimap == 128] = 1.
    alpha_f = alpha / 255.
    alpha_f = alpha_f.cuda()
    alpha_f = alpha_f.clone()*weighted
    predict = predict.clone()*weighted
    gauss_kernel = build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=True)
    pyr_alpha  = laplacian_pyramid(alpha_f, gauss_kernel, 5)
    pyr_predict = laplacian_pyramid(predict, gauss_kernel, 5)
    laplacian_loss_weighted = sum(fnn.l1_loss(a, b) for a, b in zip(pyr_alpha, pyr_predict))
    return laplacian_loss_weighted

def get_laplacian_loss_whole_img(predict, alpha):
    alpha_f = alpha / 255.
    alpha_f = alpha_f.cuda()
    gauss_kernel = build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=True)
    pyr_alpha  = laplacian_pyramid(alpha_f, gauss_kernel, 5)
    pyr_predict = laplacian_pyramid(predict, gauss_kernel, 5)
    laplacian_loss = sum(fnn.l1_loss(a, b) for a, b in zip(pyr_alpha, pyr_predict))
    return laplacian_loss

def get_composition_loss_whole_img(img, alpha, fg, bg, predict):
    weighted = torch.ones(alpha.shape).cuda()
    predict_3 = torch.cat((predict, predict, predict), 1)
    comp = predict_3 * fg + (1. - predict_3) * bg
    comp_loss = torch.sqrt((comp - img) ** 2 + 1e-12)
    comp_loss = comp_loss.sum()/(weighted.sum())
    return comp_loss

##############################
### Test loss for matting
##############################
def calculate_sad_mse_mad(predict_old,alpha,trimap):
    predict = np.copy(predict_old)
    pixel = float((trimap == 128).sum())
    predict[trimap == 255] = 1.
    predict[trimap == 0  ] = 0.
    sad_diff = np.sum(np.abs(predict - alpha))/1000
    if pixel==0:
        pixel = trimap.shape[0]*trimap.shape[1]-float((trimap==255).sum())-float((trimap==0).sum())
    mse_diff = np.sum((predict - alpha) ** 2)/pixel
    mad_diff = np.sum(np.abs(predict - alpha))/pixel
    return sad_diff, mse_diff, mad_diff
    
def calculate_sad_mse_mad_whole_img(predict, alpha):
    pixel = predict.shape[0]*predict.shape[1]
    sad_diff = np.sum(np.abs(predict - alpha))/1000
    mse_diff = np.sum((predict - alpha) ** 2)/pixel
    mad_diff = np.sum(np.abs(predict - alpha))/pixel
    return sad_diff, mse_diff, mad_diff 

def calculate_sad_fgbg(predict, alpha, trimap):
    sad_diff = np.abs(predict-alpha)
    weight_fg = np.zeros(predict.shape)
    weight_bg = np.zeros(predict.shape)
    weight_trimap = np.zeros(predict.shape)
    weight_fg[trimap==255] = 1.
    weight_bg[trimap==0  ] = 1.
    weight_trimap[trimap==128  ] = 1.
    sad_fg = np.sum(sad_diff*weight_fg)/1000
    sad_bg = np.sum(sad_diff*weight_bg)/1000
    sad_trimap = np.sum(sad_diff*weight_trimap)/1000
    return sad_fg, sad_bg

def compute_gradient_whole_image(pd, gt):
    from scipy.ndimage import gaussian_filter
    pd_x = gaussian_filter(pd, sigma=1.4, order=[1, 0], output=np.float32)
    pd_y = gaussian_filter(pd, sigma=1.4, order=[0, 1], output=np.float32)
    gt_x = gaussian_filter(gt, sigma=1.4, order=[1, 0], output=np.float32)
    gt_y = gaussian_filter(gt, sigma=1.4, order=[0, 1], output=np.float32)
    pd_mag = np.sqrt(pd_x**2 + pd_y**2)
    gt_mag = np.sqrt(gt_x**2 + gt_y**2)

    error_map = np.square(pd_mag - gt_mag)
    loss = np.sum(error_map) / 10
    return loss

def compute_connectivity_loss_whole_image(pd, gt, step=0.1):

    from scipy.ndimage import morphology
    from skimage.measure import label, regionprops
    h, w = pd.shape
    thresh_steps = np.arange(0, 1.1, step)
    l_map = -1 * np.ones((h, w), dtype=np.float32)
    lambda_map = np.ones((h, w), dtype=np.float32)
    for i in range(1, thresh_steps.size):
        pd_th = pd >= thresh_steps[i]
        gt_th = gt >= thresh_steps[i]
        label_image = label(pd_th & gt_th, connectivity=1)
        cc = regionprops(label_image)
        size_vec = np.array([c.area for c in cc])
        if len(size_vec) == 0:
            continue
        max_id = np.argmax(size_vec)
        coords = cc[max_id].coords
        omega = np.zeros((h, w), dtype=np.float32)
        omega[coords[:, 0], coords[:, 1]] = 1
        flag = (l_map == -1) & (omega == 0)
        l_map[flag == 1] = thresh_steps[i-1]
        dist_maps = morphology.distance_transform_edt(omega==0)
        dist_maps = dist_maps / dist_maps.max()
    l_map[l_map == -1] = 1
    d_pd = pd - l_map
    d_gt = gt - l_map
    phi_pd = 1 - d_pd * (d_pd >= 0.15).astype(np.float32)
    phi_gt = 1 - d_gt * (d_gt >= 0.15).astype(np.float32)
    loss = np.sum(np.abs(phi_pd - phi_gt)) / 1000
    return loss



def gen_trimap_from_segmap_e2e(segmap):
    trimap = np.argmax(segmap, axis=1)[0]
    trimap = trimap.astype(np.int64)    
    trimap[trimap==1]=128
    trimap[trimap==2]=255
    return trimap.astype(np.uint8)

def get_masked_local_from_global(global_sigmoid, local_sigmoid):
    values, index = torch.max(global_sigmoid,1)
    index = index[:,None,:,:].float()
    ### index <===> [0, 1, 2]
    ### bg_mask <===> [1, 0, 0]
    bg_mask = index.clone()
    bg_mask[bg_mask==2]=1
    bg_mask = 1- bg_mask
    ### trimap_mask <===> [0, 1, 0]
    trimap_mask = index.clone()
    trimap_mask[trimap_mask==2]=0
    ### fg_mask <===> [0, 0, 1]
    fg_mask = index.clone()
    fg_mask[fg_mask==1]=0
    fg_mask[fg_mask==2]=1
    fusion_sigmoid = local_sigmoid*trimap_mask+fg_mask
    return fusion_sigmoid

def get_masked_local_from_global_test(global_result, local_result):
    weighted_global = np.ones(global_result.shape)
    weighted_global[global_result==255] = 0
    weighted_global[global_result==0] = 0
    fusion_result = global_result*(1.-weighted_global)/255+local_result*weighted_global
    return fusion_result
def inference_once( model, scale_img, scale_trimap=None):
    pred_list = []
    tensor_img = torch.from_numpy(scale_img[:, :, :]).permute(2, 0, 1).cuda()
    input_t = tensor_img
    input_t = input_t/255.0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    input_t = normalize(input_t)
    input_t = input_t.unsqueeze(0).float()
   # pred_global, pred_local, pred_fusion = model(input_t)[:3]
    pred_fusion = model(input_t)[:3]
    pred_global = pred_fusion
    pred_local = pred_fusion

    pred_global = pred_global.data.cpu().numpy()
    pred_global = gen_trimap_from_segmap_e2e(pred_global)
    pred_local = pred_local.data.cpu().numpy()[0,0,:,:]
    pred_fusion = pred_fusion.data.cpu().numpy()[0,0,:,:]
    return pred_global, pred_local, pred_fusion

# def inference_img( test_choice,model, img):
#     h, w, c = img.shape
#     new_h = min(config['datasets'].MAX_SIZE_H, h - (h % 32))
#     new_w = min(config['datasets'].MAX_SIZE_W, w - (w % 32))
#     if test_choice=='HYBRID':
#         global_ratio = 1/2
#         local_ratio = 1
#         resize_h = int(h*global_ratio)
#         resize_w = int(w*global_ratio)
#         new_h = min(config['datasets'].MAX_SIZE_H, resize_h - (resize_h % 32))
#         new_w = min(config['datasets'].MAX_SIZE_W, resize_w - (resize_w % 32))
#         scale_img = resize(img,(new_h,new_w))*255.0
#         pred_coutour_1, pred_retouching_1, pred_fusion_1 = inference_once( model, scale_img)
#         pred_coutour_1 = resize(pred_coutour_1,(h,w))*255.0
#         resize_h = int(h*local_ratio)
#         resize_w = int(w*local_ratio)
#         new_h = min(config['datasets'].MAX_SIZE_H, resize_h - (resize_h % 32))
#         new_w = min(config['datasets'].MAX_SIZE_W, resize_w - (resize_w % 32))
#         scale_img = resize(img,(new_h,new_w))*255.0
#         pred_coutour_2, pred_retouching_2, pred_fusion_2 = inference_once( model, scale_img)        
#         pred_retouching_2 = resize(pred_retouching_2,(h,w))
#         pred_fusion = get_masked_local_from_global_test(pred_coutour_1, pred_retouching_2)
#         return pred_fusion
#     else:
#         resize_h = int(h/2)
#         resize_w = int(w/2)
#         new_h = min(config['datasets'].MAX_SIZE_H, resize_h - (resize_h % 32))
#         new_w = min(config['datasets'].MAX_SIZE_W, resize_w - (resize_w % 32))
#         scale_img = resize(img,(new_h,new_w))*255.0
#         pred_global, pred_local, pred_fusion = inference_once( model, scale_img)
#         pred_local = resize(pred_local,(h,w))
#         pred_global = resize(pred_global,(h,w))*255.0
#         pred_fusion = resize(pred_fusion,(h,w))
#         return pred_fusion


def inference_img(model, img):
    h,w,_ = img.shape
    # print(img.shape)
    if h%8!=0 or w%8!=0:
        img=cv2.copyMakeBorder(img, 8-h%8, 0, 8-w%8, 0, cv2.BORDER_REFLECT)
    # print(img.shape)

    tensor_img = torch.from_numpy(img).permute(2, 0, 1).cuda()
    input_t = tensor_img
    input_t = input_t/255.0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    input_t = normalize(input_t)
    input_t = input_t.unsqueeze(0).float()
    with torch.no_grad():
        out=model(input_t)
    # print("out",out.shape)
    result = out[0][:,-h:,-w:].cpu().numpy()
    # print(result.shape)

    return result[0]
    
    

def test_am2k(model):
    ############################
    # Some initial setting for paths
    ############################
    ORIGINAL_PATH = config['datasets']['am2k']['validation_original']
    MASK_PATH = config['datasets']['am2k']['validation_mask']
    TRIMAP_PATH = config['datasets']['am2k']['validation_trimap']
    img_paths = glob.glob(ORIGINAL_PATH+"/*.jpg")

    ############################
    # Start testing
    ############################
    sad_diffs = 0.
    mse_diffs = 0.
    mad_diffs = 0.
    grad_diffs = 0.
    conn_diffs = 0.
    sad_trimap_diffs = 0.
    mse_trimap_diffs = 0.
    mad_trimap_diffs = 0.
    sad_fg_diffs = 0.
    sad_bg_diffs = 0.


    total_number = len(img_paths)
    log("===============================")
    log(f'====> Start Testing\n\t--Dataset: AM2k\n\t-\n\t--Number: {total_number}')

    for img_path in tqdm.tqdm(img_paths):
        img_name=(img_path.split("/")[-1])[:-4]
        alpha_path = MASK_PATH+img_name+'.png'
        trimap_path = TRIMAP_PATH+img_name+'.png'
        pil_img = Image.open(img_path)
        img = np.array(pil_img)
        trimap = np.array(Image.open(trimap_path))
        alpha = np.array(Image.open(alpha_path))/255.
        img = img[:,:,:3] if img.ndim>2 else img
        trimap = trimap[:,:,0] if trimap.ndim>2 else trimap
        alpha = alpha[:,:,0] if alpha.ndim>2 else alpha

        with torch.no_grad():
            torch.cuda.empty_cache()
            predict = inference_img( model, img)

        
            sad_trimap_diff, mse_trimap_diff, mad_trimap_diff = calculate_sad_mse_mad(predict, alpha, trimap)
            sad_diff, mse_diff, mad_diff = calculate_sad_mse_mad_whole_img(predict, alpha)
            sad_fg_diff, sad_bg_diff = calculate_sad_fgbg(predict, alpha, trimap)
            conn_diff = compute_connectivity_loss_whole_image(predict, alpha)
            grad_diff = compute_gradient_whole_image(predict, alpha)

            log(f"[{img_paths.index(img_path)}/{total_number}]\nImage:{img_name}\nsad:{sad_diff}\nmse:{mse_diff}\nmad:{mad_diff}\nsad_trimap:{sad_trimap_diff}\nmse_trimap:{mse_trimap_diff}\nmad_trimap:{mad_trimap_diff}\nsad_fg:{sad_fg_diff}\nsad_bg:{sad_bg_diff}\nconn:{conn_diff}\ngrad:{grad_diff}\n-----------")

            sad_diffs += sad_diff
            mse_diffs += mse_diff
            mad_diffs += mad_diff
            mse_trimap_diffs += mse_trimap_diff
            sad_trimap_diffs += sad_trimap_diff
            mad_trimap_diffs += mad_trimap_diff
            sad_fg_diffs += sad_fg_diff
            sad_bg_diffs += sad_bg_diff
            conn_diffs += conn_diff
            grad_diffs += grad_diff
            Image.fromarray(np.uint8(predict*255)).save(f"test/{img_name}.png")


    log("===============================")
    log(f"Testing numbers: {total_number}")

   
    log("SAD: {}".format(sad_diffs / total_number))
    log("MSE: {}".format(mse_diffs / total_number))
    log("MAD: {}".format(mad_diffs / total_number))
    log("GRAD: {}".format(grad_diffs / total_number))
    log("CONN: {}".format(conn_diffs / total_number))
    log("SAD TRIMAP: {}".format(sad_trimap_diffs / total_number))
    log("MSE TRIMAP: {}".format(mse_trimap_diffs / total_number))
    log("MAD TRIMAP: {}".format(mad_trimap_diffs / total_number))
    log("SAD FG: {}".format(sad_fg_diffs / total_number))
    log("SAD BG: {}".format(sad_bg_diffs / total_number))
    return sad_diffs/total_number,mse_diffs/total_number,grad_diffs/total_number


def test_p3m10k(model,dataset_choice, max_image=-1):
    ############################
    # Some initial setting for paths
    ############################
    if dataset_choice == 'P3M_500_P':
        val_option = 'VAL500P'
    else:
        val_option = 'VAL500NP'
    ORIGINAL_PATH = config['datasets']['p3m10k']+"/validation/"+config['datasets']['p3m10k_test'][val_option]['ORIGINAL_PATH']
    MASK_PATH = config['datasets']['p3m10k']+"/validation/"+config['datasets']['p3m10k_test'][val_option]['MASK_PATH']
    TRIMAP_PATH = config['datasets']['p3m10k']+"/validation/"+config['datasets']['p3m10k_test'][val_option]['TRIMAP_PATH']
    ############################
    # Start testing
    ############################
    sad_diffs = 0.
    mse_diffs = 0.
    mad_diffs = 0.
    sad_trimap_diffs = 0.
    mse_trimap_diffs = 0.
    mad_trimap_diffs = 0.
    sad_fg_diffs = 0.
    sad_bg_diffs = 0.
    conn_diffs = 0.
    grad_diffs = 0.
    model.eval()
    img_paths = glob.glob(ORIGINAL_PATH+"/*.jpg")
    if (max_image>1):
        img_paths = img_paths[:max_image]
    total_number = len(img_paths)
    log("===============================")
    log(f'====> Start Testing\n\t----Test: {dataset_choice}\n\t--Number: {total_number}')

    for img_path in tqdm.tqdm(img_paths):
        img_name=(img_path.split("/")[-1])[:-4]
        alpha_path = MASK_PATH+img_name+'.png'
        trimap_path = TRIMAP_PATH+img_name+'.png'
        pil_img = Image.open(img_path)
        img = np.array(pil_img)
    
        trimap = np.array(Image.open(trimap_path))
        alpha = np.array(Image.open(alpha_path))/255.
        img = img[:,:,:3] if img.ndim>2 else img
        trimap = trimap[:,:,0] if trimap.ndim>2 else trimap
        alpha = alpha[:,:,0] if alpha.ndim>2 else alpha
        with torch.no_grad():
            torch.cuda.empty_cache()
            start = time.time()


            predict = inference_img( model, img) #HYBRID show less accuracy
        
            # tensorimg=transforms.ToTensor()(pil_img)
            # input_img=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensorimg)
          
            # predict = model(input_img.unsqueeze(0).to(device))[0][0].detach().cpu().numpy()
            # if predict.shape!=(pil_img.height,pil_img.width):
            #     print("resize for ",img_path)
            #     predict = resize(predict,(pil_img.height,pil_img.width))
            sad_trimap_diff, mse_trimap_diff, mad_trimap_diff = calculate_sad_mse_mad(predict, alpha, trimap)
            sad_diff, mse_diff, mad_diff = calculate_sad_mse_mad_whole_img(predict, alpha)
          
            sad_fg_diff, sad_bg_diff = calculate_sad_fgbg(predict, alpha, trimap)
            conn_diff = compute_connectivity_loss_whole_image(predict, alpha)
            grad_diff = compute_gradient_whole_image(predict, alpha)
            log(f"[{img_paths.index(img_path)}/{total_number}]\nImage:{img_name}\nsad:{sad_diff}\nmse:{mse_diff}\nmad:{mad_diff}\nconn:{conn_diff}\ngrad:{grad_diff}\n-----------")
            sad_diffs += sad_diff
            mse_diffs += mse_diff
            mad_diffs += mad_diff
            mse_trimap_diffs += mse_trimap_diff
            sad_trimap_diffs += sad_trimap_diff
            mad_trimap_diffs += mad_trimap_diff
            sad_fg_diffs += sad_fg_diff
            sad_bg_diffs += sad_bg_diff
            conn_diffs += conn_diff
            grad_diffs += grad_diff
    
            Image.fromarray(np.uint8(predict*255)).save(f"test/{img_name}.png")
           
    log("===============================")
    log(f"Testing numbers: {total_number}")
    log("SAD: {}".format(sad_diffs / total_number))
    log("MSE: {}".format(mse_diffs / total_number))
    log("MAD: {}".format(mad_diffs / total_number))
    log("SAD TRIMAP: {}".format(sad_trimap_diffs / total_number))
    log("MSE TRIMAP: {}".format(mse_trimap_diffs / total_number))
    log("MAD TRIMAP: {}".format(mad_trimap_diffs / total_number))
    log("SAD FG: {}".format(sad_fg_diffs / total_number))
    log("SAD BG: {}".format(sad_bg_diffs / total_number))
    log("CONN: {}".format(conn_diffs / total_number))
    log("GRAD: {}".format(grad_diffs / total_number))

    return sad_diffs/total_number,mse_diffs/total_number,grad_diffs/total_number

def log(str):
    print(str)
    logging.info(str)

if __name__ == '__main__':
    print('*********************************')  
    config = OmegaConf.load(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "config/base.yaml"))
    config=OmegaConf.merge(config,OmegaConf.from_cli())
    print(config)
    model =  StyleMatte()
    model =  model.to(device)
    checkpoint = f"{config.checkpoint_dir}/{config.checkpoint}"
    state_dict = torch.load(checkpoint, map_location=f'{device}')
    print("loaded",checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    logging.basicConfig(filename=f'report/{config.checkpoint.replace("/","--")}.report', encoding='utf-8',filemode='w', level=logging.INFO)
    # ckpt = torch.load("checkpoints/p3mnet_pretrained_on_p3m10k.pth")
    # model.load_state_dict(ckpt['state_dict'], strict=True)
    # model = model.cuda()
    if config.dataset_to_use =="AM2K":
         test_am2k(model)
    else:
        for dataset_choice in ['P3M_500_P','P3M_500_NP']:        
            test_p3m10k(model,dataset_choice)

