import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from .module_util import SinusoidalPosEmb, LayerNorm, exists
from .fusion import Fusion

import matplotlib.pyplot as plt

import torch
import torchvision.ops as ops
from torch import nn
import numpy as np



def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col
 
 
# def visualize_feature_map(img_batch):
#     feature_map = np.squeeze(img_batch, axis=0)
#     print(feature_map.shape)
 
#     feature_map_combination = []
#     plt.figure()
 
#     num_pic = feature_map.shape[0]
#     row, col = get_row_col(num_pic)
 
#     for i in range(0, num_pic):
#         feature_map_split = feature_map[i, :, :]
#         feature_map_combination.append(feature_map_split)
#         plt.subplot(row, col, i + 1)
#         plt.subplots_adjust(hspace=0, wspace=0)
        
#         plt.imshow(feature_map_split)
#         # axis('off')
#         plt.axis('off')
#         # title('feature_map_{}'.format(i))
 
#     plt.savefig('feature_map.png')
#     plt.show()
 
    # 各个特征图按1：1 叠加
    # feature_map_sum = sum(ele for ele in feature_map_combination)
    # plt.imshow(feature_map_sum)
    # plt.show()
    # plt.savefig("feature_map_sum.png")




# class DCN(nn.Module):
#     def __init__(self, in_c, out_c, k=3):
#         super().__init__()
#         p = (k - 1) // 2
#         self.split_size = (2 * k * k, k * k)
#         self.conv_offset = nn.Conv2d(in_c, 3 * k * k, k, padding=p)
#         self.conv_deform = ops.DeformConv2d(in_c, out_c, k, padding=p)

#         # initialize
#         nn.init.constant_(self.conv_offset.weight, 0)
#         nn.init.constant_(self.conv_offset.bias, 0)
#         nn.init.kaiming_normal_(self.conv_deform.weight, mode='fan_out', nonlinearity='relu')

#     def forward(self, x):
#         offset, mask = torch.split(self.conv_offset(x), self.split_size, dim=1)
#         mask = torch.sigmoid(mask)
#         y = self.conv_deform(x, offset, mask)
#         return y


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


# class NAFBlock(nn.Module):
#     def __init__(self, c, time_emb_dim=None, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             SimpleGate(), nn.Linear(time_emb_dim // 2, c * 4)
#         ) if time_emb_dim else None

#         dw_channel = c * DW_Expand
#         self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=True)
#         self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
#                                groups=dw_channel,
#                                bias=True)
#         self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
#                                groups=1, bias=True)

#         # Simplified Channel Attention
#         self.sca = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
#                       groups=1, bias=True),
#         )

#         # SimpleGate
#         self.sg = SimpleGate()

#         ffn_channel = FFN_Expand * c
#         self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=True)
#         self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
#                                groups=1, bias=True)

#         self.norm1 = LayerNorm(c)
#         self.norm2 = LayerNorm(c)

#         self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
#         self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

#         self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
#         self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

#     def time_forward(self, time, mlp):
#         time_emb = mlp(time)
#         time_emb = rearrange(time_emb, 'b c -> b c 1 1')
#         return time_emb.chunk(4, dim=1)

#     def forward(self, x):
#         inp, time = x
#         shift_att, scale_att, shift_ffn, scale_ffn = self.time_forward(time, self.mlp)

#         x = inp

#         x = self.norm1(x)
#         x = x * (scale_att + 1) + shift_att
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.sg(x)
#         x = x * self.sca(x)
#         x = self.conv3(x)

#         x = self.dropout1(x)

#         y = inp + x * self.beta

#         x = self.norm2(y)
#         x = x * (scale_ffn + 1) + shift_ffn
#         x = self.conv4(x)
#         x = self.sg(x)
#         x = self.conv5(x)

#         x = self.dropout2(x)

#         x = y + x * self.gamma

#         return x, time

class CFBlock(nn.Module):
    def __init__(self, c, time_emb_dim=None, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            SimpleGate(), nn.Linear(time_emb_dim // 2, c * 4)
        ) if time_emb_dim else None

        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        # self.conv2 = DCN(dw_channel, dw_channel)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               bias=True, groups=dw_channel, )
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel // 2, bias=True)

        self.dconv1 = nn.Sequential(
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=3, padding=2, stride=1,
                                dilation=2, groups=dw_channel // 2,
                                bias=True),
            nn.GELU(),
            # nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2 ,kernel_size=1, padding=0, stride=1)
            # SimpleGate()
        )

        self.dconv2 = nn.Sequential(

            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=3, padding=4, stride=1,
                                dilation=4, groups=dw_channel // 2,
                                bias=True),
            nn.GELU(),
            # nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2,kernel_size=1, padding=0, stride=1)
        )


        self.dconv3 = nn.Sequential(
             nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=3, padding=8, stride=1,
                                dilation=8, groups=dw_channel // 2,
                                bias=True),
            nn.GELU(),
            # nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2,kernel_size=1, padding=0, stride=1)
            )

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.fusion = Fusion(dw_channel // 2)
        self.gelu = nn.GELU()
    def time_forward(self, time, mlp):
        time_emb = mlp(time)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        return time_emb.chunk(4, dim=1)

    def forward(self, x):
        inp, time = x
        shift_att, scale_att, shift_ffn, scale_ffn = self.time_forward(time, self.mlp)

        x = inp

        x = self.norm1(x)
        x = x * (scale_att + 1) + shift_att
        x = self.conv1(x)

        x = self.conv2(x)

        # print(x1.shape, x.shape)

        x = self.sg(x)
        x = x * self.sca(x)
        x1 = self.dconv1(x)
        x2 = self.dconv2(x1)
        x3 = self.dconv3(x2)
        # input()
        # visualize_feature_map(x.clone().cpu())
        # input()
        # visualize_feature_map(x3.clone().cpu())
        x = self.fusion(x,x3)
        # x + x3
        # visualize_feature_map(x.clone().cpu())
        # input()
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = x * (scale_ffn + 1) + shift_ffn
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        x = y + x * self.gamma

        return x, time


class ConditionalNAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], upscale=1):
        super().__init__()
        self.upscale = upscale
        fourier_dim = width
        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)
        time_dim = width * 4

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim * 2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim)
        )

        self.intro = nn.Conv2d(in_channels=img_channel * 2, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[CFBlock(chan, time_dim) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[CFBlock(chan, time_dim) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[CFBlock(chan, time_dim) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, cond, time):
        inp_res = inp.clone()

        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(inp.device)

        x = inp - cond
        x = torch.cat([x, cond], dim=1)

        t = self.time_mlp(time)

        B, C, H, W = x.shape
        x = self.check_image_size(x)

        x = self.intro(x)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x, _ = encoder([x, t])
            encs.append(x)
            x = down(x)

        x, _ = self.middle_blks([x, t])

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x, _ = decoder([x, t])

        x = self.ending(x)

        x = x[..., :H, :W]

        return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

