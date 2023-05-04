import cv2
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from itertools import chain

from transformers import Mask2FormerForUniversalSegmentation # we use 
from dataset import inv_normalize


def conv2d_relu(input_filters,output_filters,kernel_size=3,  bias=True):
    return nn.Sequential(
        nn.Conv2d(input_filters, output_filters, kernel_size=kernel_size, padding=kernel_size//2, bias=bias),
        nn.LeakyReLU(0.2, inplace=True),
        nn.BatchNorm2d(output_filters)
    )

def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y

class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                                    for ft_size in feature_channels[1:]])
        self.smooth_conv =  nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)] 
                                    * (len(feature_channels)-1))
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(2*fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        
        features[:-1] = [conv1x1(feature) for feature, conv1x1 in zip(features[:-1], self.conv1x1)]##
        feature=up_and_add(self.smooth_conv[0](features[0]),features[1])
        feature=up_and_add(self.smooth_conv[1](feature),features[2])
        feature=up_and_add(self.smooth_conv[2](feature),features[3])


        H, W = features[-1].size(2), features[-1].size(3)
        x = [feature,features[-1]]
        x = [F.interpolate(x_el, size=(H, W), mode='bilinear', align_corners=True) for x_el in x]

        x = self.conv_fusion(torch.cat(x, dim=1))
        #x = F.interpolate(x, size=(H*4, W*4), mode='bilinear', align_corners=True) 
        return x

class ConvGuidedFilter(nn.Module):
    def __init__(self, radius=1):
        super(ConvGuidedFilter, self).__init__()

        self.box_filter = nn.Conv2d(1, 1, kernel_size=3, padding=radius, dilation=radius, bias=False)
        self.box_filter.weight.data[...] = 1.0
       # self.box_filter = BoxFilter(radius)


        self.conv_a = nn.Sequential(nn.Conv2d(2, 32, kernel_size=3, bias=False),
                                    nn.BatchNorm2d(32),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(32, 32, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(32),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(32, 1, kernel_size=3, bias=False))
        torch.nn.init.dirac_(self.conv_a.weight)

        

    def forward(self, x_lr, y_lr, x_hr):
        _, _, h_lrx, w_lrx = x_lr.size()
        _, _, h_hrx, w_hrx = x_hr.size()

        N = self.box_filter(x_lr.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0))
        ## mean_x
        mean_x = self.box_filter(x_lr)/N
        ## mean_y
        mean_y = self.box_filter(y_lr)/N
        ## cov_xy
        cov_xy = self.box_filter(x_lr * y_lr)/N - mean_x * mean_y
        ## var_x
        var_x  = self.box_filter(x_lr * x_lr)/N - mean_x * mean_x

        ## A
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return mean_A * x_hr + mean_b

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class StyleMatte(nn.Module):
    def __init__(self,fpn_fuse=True,single_conv=True):
        super(StyleMatte, self).__init__()
        self.fpn = FPN_fuse(feature_channels=[256, 256, 256, 256],fpn_out=256)
        self.pixel_decoder =  Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-coco-instance").base_model.pixel_level_module
        self.fgf = FastGuidedFilter()
        if single_conv:
            self.conv = nn.Conv2d(256,1,kernel_size=3,padding=1)
        else:
            self.conv =  self.conv = nn.Sequential(nn.Conv2d(256, 16, kernel_size=1),
                                        nn.BatchNorm2d(16),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(16, 1, kernel_size=3,padding=1))
    def forward(self, image, normalize=False):
        # if normalize:
        #     image.sub_(self.get_buffer("image_net_mean")).div_(self.get_buffer("image_net_std"))
        
        decoder_out = self.pixel_decoder(image)  
        decoder_states=list(decoder_out.decoder_hidden_states)
        decoder_states.append(decoder_out.decoder_last_hidden_state)    
        out=self.fpn(decoder_states)
        image_hr = inv_normalize(image).clip(0,1).mean(1, keepdim=True)
        
        image_lr=nn.functional.interpolate(image_hr,
                scale_factor=0.25,
                mode='bicubic',
                align_corners=True
               )
        mask_lr = self.conv(out)
        mask_result = self.fgf(image_lr,mask_lr,image_hr)
      
        # out =  nn.functional.interpolate(out,
                            # scale_factor=4,
                            # mode='bicubic',
                            # align_corners=True
                        # )
        
        return mask_result.clip(0,1)#torch.sigmoid(mask_result)
    def get_training_params(self):
        return list(self.fpn.parameters())+list(self.conv.parameters())+list(self.pixel_decoder.decoder.parameters())
    

class ConvGuidedFilter(nn.Module):
    def __init__(self, radius=1):
        super(ConvGuidedFilter, self).__init__()

        self.box_filter = nn.Conv2d(1, 1, kernel_size=3, padding=radius, dilation=radius, bias=False)
        self.box_filter.weight.data[...] = 1.0
       # self.box_filter = BoxFilter(radius)


        self.conv_a = nn.Sequential(nn.Conv2d(2, 32, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(32),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(32, 32, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(32),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(32, 1, kernel_size=1, bias=False))

    def forward(self, x_lr, y_lr, x_hr):
        _, _, h_lrx, w_lrx = x_lr.size()
        _, _, h_hrx, w_hrx = x_hr.size()

        N = self.box_filter(x_lr.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0))
        ## mean_x
        mean_x = self.box_filter(x_lr)/N
        ## mean_y
        mean_y = self.box_filter(y_lr)/N
        ## cov_xy
        cov_xy = self.box_filter(x_lr * y_lr)/N - mean_x * mean_y
        ## var_x
        var_x  = self.box_filter(x_lr * x_lr)/N - mean_x * mean_x

        ## A
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return mean_A * x_hr + mean_b
class FastGuidedFilter(nn.Module):
    def __init__(self, r=1, eps=1e-8):
        super(FastGuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)


    def forward(self, lr_x, lr_y, hr_x):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2*self.r+1 and w_lrx > 2*self.r+1

        ## N
        N = self.boxfilter(lr_x.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0))

        ## mean_x
        mean_x = self.boxfilter(lr_x) / N
        ## mean_y
        mean_y = self.boxfilter(lr_y) / N
        ## cov_xy
        cov_xy = self.boxfilter(lr_x * lr_y) / N - mean_x * mean_y
        ## var_x
        var_x = self.boxfilter(lr_x * lr_x) / N - mean_x * mean_x

        ## A
        A = cov_xy / (var_x + self.eps)
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return mean_A*hr_x+mean_b

def diff_x(input, r):
    assert input.dim() == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output

def diff_y(input, r):
    assert input.dim() == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output

class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def forward(self, x):
        assert x.dim() == 4

        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)
    
if __name__ == '__main__':
    model = StyleMatte().cuda()
    out=model(torch.randn(1,3,640,480).cuda())
    print(out.shape)
