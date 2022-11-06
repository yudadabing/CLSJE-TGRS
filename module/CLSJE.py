import torch.nn as nn
import torch.nn.functional as F
from simplecv.interface import CVModule
from simplecv import registry
import torch
from torch.autograd import Variable
import numpy as np
import math
import scipy.io as io
from simplecv import dp_train as train
import matplotlib.pyplot as plt
import cv2
import os

args = train.parser.parse_args()
dataset_path=args.config_path

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

def conv3x3_gn_relu(in_channel, out_channel, num_group):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 1, 1),
        nn.GroupNorm(num_group, out_channel),
        nn.ReLU(inplace=True),
    )

def gn_relu(in_channel, num_group):
    return nn.Sequential(
        nn.GroupNorm(num_group, in_channel),
        nn.ReLU(inplace=True),
    )


def downsample2x(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 2, 1),
        nn.ReLU(inplace=True)
    )

class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x3,x4):
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv(x)

def repeat_block(block_channel1, r, n):
    cl_channel = block_channel1 / 8
    cl_channel = int(cl_channel)
    cl2_channel = int(cl_channel / 2)
    gn_a = int(block_channel1 / 2)
    layers = [
        nn.Sequential(
            BasicConv(block_channel1,gn_a, kernel_size=3, stride=1, relu=True), 
            RCAB_last2(gn_a), gn_relu(block_channel1, r)
            , )]
    return nn.Sequential(*layers)

class SCM(nn.Module):
## HOS 144
##BS 145
## IP 200
##PAVIA 103

    def __init__(self,out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(145, out_plane // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane+145, out_plane, kernel_size=1, stride=1, relu=False)
        



    def forward(self, x):
        y=self.main(x)
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)

class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out

## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat):
        super(SAM, self).__init__()
        self.conv1 = BasicConv(n_feat, n_feat,  kernel_size=3, stride=1, relu=False)
        self.conv2 = BasicConv(n_feat, n_feat, kernel_size=3, stride=1, relu=False)
        # self.conv3 = BasicConv(n_feat, n_feat, kernel_size=3, stride=1, relu=False)

    def forward(self, x, x_img):
        img = self.conv2(x) + self.conv1(x_img)
        x2 = torch.sigmoid(img)
        x1 = x*x2
        x1 = x1+x
        return x1




@registry.MODEL.register('CLSJE')
class CLSJE(CVModule):
    def __init__(self, config):
        super(CLSJE, self).__init__(config)
        r = int(4 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        block_total=block1_channels+block2_channels+block3_channels+block4_channels
        self.feature_ops1 = nn.Sequential(
            conv3x3_gn_relu(self.config.in_channels, block1_channels, r),
            repeat_block(block1_channels, r, self.config.num_blocks[0]))# num_blocks=(1, 1, 1, 1)


        self.feature_ops2= nn.Sequential(
            downsample2x(block1_channels, block2_channels),
            repeat_block(block2_channels, r, self.config.num_blocks[1]))

        self.feature_ops3= nn.Sequential(
            downsample2x(block2_channels, block3_channels),
            repeat_block(block3_channels, r, self.config.num_blocks[2]))
            # nn.Identity(),

        self.feature_ops4= nn.Sequential(
            downsample2x(block3_channels, block4_channels),
            repeat_block(block4_channels, r, self.config.num_blocks[3]),

        )


        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
        self.feat_extract = nn.ModuleList([
            # BasicConv(self.config.in_channels, block1_channels, kernel_size=3, relu=True, stride=1),
            BasicConv(block1_channels, block1_channels , kernel_size=3, relu=True, stride=1),
            BasicConv(block2_channels, block3_channels, kernel_size=3, relu=True, stride=1),
            BasicConv(block3_channels,block4_channels, kernel_size=4, relu=True, stride=1, transpose=True),
            BasicConv(block4_channels, 145, kernel_size=4, relu=True, stride=2, transpose=True),
            # BasicConv(base_channel, 103, kernel_size=3, relu=False, stride=1)
        ])

        self.reduce_1x1convs = nn.ModuleList([
            nn.Conv2d(block1_channels, inner_dim, 1),
            nn.Conv2d(block2_channels, inner_dim, 1),
            nn.Conv2d(block3_channels, inner_dim, 1),
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        self.fuse_3x3convs = nn.ModuleList([
            conv3x3_gn_relu(inner_dim, inner_dim, r),
            conv3x3_gn_relu(inner_dim, inner_dim, r),
            conv3x3_gn_relu(inner_dim, inner_dim, r),
            nn.Conv2d(inner_dim, self.config.in_channels, 3, 1, 1),
        ])

        self.fuse_3x3convs2= nn.ModuleList([
            conv3x3_gn_relu(inner_dim, inner_dim, r),
            conv3x3_gn_relu(inner_dim*2, inner_dim, r),
            conv3x3_gn_relu(inner_dim*2, inner_dim, r),
            nn.Conv2d(inner_dim*2, self.config.in_channels, 3, 1, 1),
        ])        
        self.AFFs = nn.ModuleList([
            AFF(block_total , inner_dim),
            AFF(block_total, inner_dim),
            AFF(block_total, inner_dim),
        ])        
        

        self.FAM1 = SAM(block1_channels)
        self.SCM1 = SCM(block1_channels )
        self.FAM2 = SAM(block2_channels)
        self.SCM2 = SCM(block2_channels )
        self.FAM3 = SAM(block3_channels)
        self.SCM3 = SCM(block3_channels )
        self.cls_pred_conv = nn.Conv2d(self.config.in_channels, self.config.num_classes, 1)

        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)
        self.drop3 = nn.Dropout2d(0.1)
    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='bilinear')

        return lateral + top2x

    def top_down2(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='bilinear')
        z = torch.cat([lateral, top2x], dim=1)
        return z
        # return lateral + top2x        

    def forward(self, x, y=None, w=None, **kwargs):
        feat_list = []

        inner_feat_list=[]
        inner_feat_list2=[]
        res1 = self.feature_ops1(x)

        feat_list.append(res1)

        z2 = self.SCM1(x)  

        z =  self.FAM1(res1, z2)
        res2 =  self.feature_ops2(z)  
        feat_list.append(res2)

        z4 = self.SCM2(x) 
        z4 =  F.interpolate(z4, scale_factor=0.5)   
        z = self.FAM2(res2, z4)
        res3 = self.feature_ops3(z)
        feat_list.append(res3)


        z8 = self.SCM3(x) 
        z8 =  F.interpolate(z8, scale_factor=0.25)   
        z = self.FAM3(res3, z8)
        res4 = self.feature_ops4(z)
        feat_list.append(res4)

        res2_1=  F.interpolate(res2, scale_factor=2)   
        res3_1=  F.interpolate(res3, scale_factor=4)   
        res4_1=  F.interpolate(res4, scale_factor=8)   

        res1_2=  F.interpolate(res1, scale_factor=0.5)   
        res3_2=  F.interpolate(res3, scale_factor=2)   
        res4_2=  F.interpolate(res4, scale_factor=4)   
        
        res1_3=  F.interpolate(res1, scale_factor=0.25)   
        res2_3=  F.interpolate(res2, scale_factor=0.5)   
        res4_3=  F.interpolate(res4, scale_factor=2)   


        cres1=self.AFFs[0](res1,res2_1,res3_1,res4_1)
        cres1 = self.drop1(cres1)
        inner_feat_list2.append(cres1)

        cres2=self.AFFs[1](res1_2,res2,res3_2,res4_2)
        cres2 = self.drop1(cres2)
        inner_feat_list2.append(cres2)

        cres3=self.AFFs[2](res1_3,res2_3,res3,res4_3)
        cres3 = self.drop1(cres3)
        inner_feat_list2.append(cres3)

        cres4 = self.reduce_1x1convs[3](feat_list[3])
        inner_feat_list2.append(cres4)
        inner_feat_list2.reverse()
        out_feat_list2 = [self.fuse_3x3convs2[0](inner_feat_list2[0])]
        for i in range(len(inner_feat_list2) - 1):
            inner = self.top_down2(out_feat_list2[i], inner_feat_list2[i + 1])

            out2 = self.fuse_3x3convs2[i + 1](inner)

            out_feat_list2.append(out2)        

        final_feat = out_feat_list2[-1]

        logit = self.cls_pred_conv(final_feat)
        if self.training:
            loss_dict = {
                'cls_loss': self.loss(logit, y, w)
            }
            return loss_dict



        return torch.softmax(logit, dim=1)


    def loss(self, x, y, weight):
        beta = 0.9999
        if dataset_path =="CLSJE.CLSJE_1_0_pavia":
            cls_num_list = [6631, 18649, 2099, 3064, 1345, 5029, 1330, 3682, 947]
        elif dataset_path =="CLSJE.CLSJE_1_0_Indianpine":
            cls_num_list = [46, 1428, 830, 237, 483, 730, 28, 478, 20, 927, 2455, 593, 205, 1265, 386, 93]
        elif dataset_path == "CLSJE.CLSJE_1_0_salinas":
            cls_num_list = [2009,3726,1976,1394,2678,3959,3579,11271,6203,3278,1068,1927,916,1070,7268,1807]
        elif dataset_path == "CLSJE.CLSJE_1_0_HOS":
            cls_num_list = [1251,1254,697,1244,1242,325,1268,1244,1252,1227,1235,1233,469,428,660]
        elif dataset_path == "CLSJE.CLSJE_1_0_BS":
            cls_num_list = [270,101,251,215,269,269,259,203,314,248,305,181,268,95]
        else:
            print("no cls_num_list")

        # effective_num = 1.0 - np.power(beta, cls_num_list)
        # per_cls_weights = (1.0 - beta) / np.array(effective_num)
        # per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        # per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
        # losses = F.cross_entropy(x, y.long() - 1, ignore_index=-1, reduction='none', weight=per_cls_weights)
        # losses = losses.mean()
        per_cls_weights = np.sum(cls_num_list)/cls_num_list
        per_cls_weights=per_cls_weights/len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
        losses = F.cross_entropy(x, y.long() - 1, ignore_index=-1, reduction='none', weight=per_cls_weights)
        losses = losses.mean()
        return losses

    def set_defalut_config(self):
        # pavia
        self.config.update(dict(
            in_channels=103,
            num_classes=9,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))








def get_pad_layer(pad_type):
    if (pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif (pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif (pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer

def conv_identity(weight, bias):
    weight.data.zero_()
    if bias is not None:
        bias.data.zero_()
    o, i, h, w = weight.shape
    y = h//2
    x = w//2
    for p in range(i):
        for q in range(o):
            if p == q:
                weight.data[q, p, :, :] = 1.0

# Dynamic high-pass filtering layer
class HPF(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3, stride=1, pad_type='reflect', group=16):
        super(HPF, self).__init__()
        self.pad = get_pad_layer(pad_type)(kernel_size//2)
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(in_channels, group*kernel_size*kernel_size, kernel_size=kernel_size, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size*kernel_size)
        self.softmax = nn.Softmax(dim=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        sigma = self.conv(self.pad(x))  # channel: 64 -> 144 (group=16 * kernel=3 * kernel=3)
        sigma = self.bn(sigma)
        sigma = self.softmax(sigma)

        n, c, h, w = sigma.shape
        sigma = sigma.reshape(n, 1, c, h*w)

        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape((n, c, self.kernel_size*self.kernel_size, h*w))

        n, c1, p, q = x.shape
        x = x.permute(1, 0, 2, 3).reshape(self.group, c1//self.group, n, p, q).permute(2, 0, 1, 3, 4)

        n, c2, p, q = sigma.shape
        sigma = sigma.permute(2, 0, 1, 3).reshape((p//(self.kernel_size*self.kernel_size), self.kernel_size*self.kernel_size, n, c2, q)).permute(2, 0, 3, 1, 4)

        # add identity kernel (for converting LPF to HPF)
        n, c3, c4, p, q = sigma.shape
        kernel_identity = torch.zeros(n, c3, c4, p, q).cuda()
        kernel_identity[:, :, :, int((self.kernel_size*self.kernel_size)/2), :] = 1

        x = torch.sum(x*(kernel_identity - sigma), dim=3).reshape(n, c1, h, w)

        return x[:, :, torch.arange(h)%self.stride==0,:][:, :, :, torch.arange(w)%self.stride==0]





def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32',
                      'pre8']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    elif 'pre' in method:
        all_pre_indices_x = [0,0,1,1,5,5,6,6]
        all_pre_indices_y = [0,1,0,1,5,6,5,6]
        mapper_x = all_pre_indices_x[:num_freq]
        mapper_y = all_pre_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight # calculate 2D DCT coefficient

        result = torch.sum(x, dim=[2,3])

        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
        return dct_filter

# Multi Spectral Attention (MSA) Layer
class MSALayer(nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super(MSALayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        
        mapper_x = [temp_x*(dct_h//7) for temp_x in mapper_x]
        mapper_y = [temp_y*(dct_w//7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        
        y = self.dct_layer(x_pooled)
        y = self.fc(y).view(n, c, 1, 1)

        return x * y.expand_as(x)

def Conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


class RCAB_last2(nn.Module):
    def __init__(self,  n_feat):
        super(RCAB_last2, self).__init__()
        kernel_size=3
        bias=True
        bn=False
        act=nn.ReLU(True)
        c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7), (48, 7)])
        modules_body1 = []
        modules_body2= []


        modules_body1.append(Conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body1.append(HPF(n_feat, kernel_size))
        self.body1 = nn.Sequential(*modules_body1)

        modules_body2.append(Conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body2.append(MSALayer(n_feat, c2wh[48], c2wh[48]))
        self.body2= nn.Sequential(*modules_body2)

    def forward(self, x):
        out = torch.cat([self.body2(x), self.body1(x)], dim=1)
        return out




