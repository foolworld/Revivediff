import torch
from torch import nn
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt
import numpy as np

def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col
 

def visualize_channel_weights(channel_weights):
    """
    可视化通道权重。
    
    参数:
    channel_weights -- 形状为 (C, 1, 1) 的通道权重矩阵
    save_path -- 保存图像文件的路径，默认为 'channel_weights.png'
    """
    if channel_weights.shape != (1, 64, 1, 1):
        raise ValueError("输入的 channel_weights 应该是形状为 (1, 64, 1, 1) 的4D张量。")
    
    # 将张量转换为一维向量
    channel_weights = channel_weights.view(64).detach().cpu().numpy()
    
    # 重塑为8x8的矩阵
    heatmap = channel_weights.reshape(8, 8)
    
    # 创建绘图窗口
    plt.figure(figsize=(6, 6))
    plt.axis("off")
    # 绘制热力图
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    for i in range(8):
        for j in range(8):
            plt.text(j, i, f'{heatmap[i, j]:.2f}', ha='center', va='center', color='white', fontsize=8)
    # plt.colorbar()  # 添加颜色条
    
    # 添加标题
    # plt.title('Channel Attention Weights Heatmap')
    
    # 保存和展示图像
    # plt.savefig(save_path)
    plt.show()
    
def visualize_feature_map(img_batch):
    feature_map = np.squeeze(img_batch, axis=0)
    print(feature_map.shape)
 
    feature_map_combination = []
    plt.figure()
 
    num_pic = feature_map.shape[0]
    row, col = get_row_col(num_pic)
 
    for i in range(0, num_pic):
        feature_map_split = feature_map[i, :, :]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.subplots_adjust(hspace=0.1, wspace=0.1)
        plt.imshow(feature_map_split)
        # axis('off')
        plt.axis('off')
        
        # title('feature_map_{}'.format(i))
 
    plt.savefig('feature_map.png')
    plt.show()
 
    # 各个特征图按1：1 叠加
    # feature_map_sum = sum(ele for ele in feature_map_combination)
    # plt.imshow(feature_map_sum)
    # plt.show()
    # plt.savefig("feature_map_sum.png")

class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect' ,bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction = 8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn





# class _NonLocalBlockND(nn.Module):
    # def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=False):
    #     super(_NonLocalBlockND, self).__init__()

    #     assert dimension in [1, 2, 3]

    #     self.dimension = dimension
    #     self.sub_sample = sub_sample

    #     self.in_channels = in_channels
    #     self.inter_channels = inter_channels

    #     if self.inter_channels is None:
    #         self.inter_channels = in_channels // 2
    #         if self.inter_channels == 0:
    #             self.inter_channels = 1

    #     if dimension == 3:
    #         conv_nd = nn.Conv3d
    #         max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
    #         bn = nn.BatchNorm3d
    #     elif dimension == 2:
    #         conv_nd = nn.Conv2d
    #         max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
    #         bn = nn.BatchNorm2d
    #     else:
    #         conv_nd = nn.Conv1d
    #         max_pool_layer = nn.MaxPool1d(kernel_size=(2))
    #         bn = nn.BatchNorm1d

    #     self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
    #                      kernel_size=1, stride=1, padding=0)

    #     if bn_layer:
    #         self.W = nn.Sequential(
    #             conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
    #                     kernel_size=1, stride=1, padding=0),
    #             bn(self.in_channels)
    #         )
    #         nn.init.constant_(self.W[1].weight, 0)
    #         nn.init.constant_(self.W[1].bias, 0)
    #     else:
    #         self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
    #                          kernel_size=1, stride=1, padding=0)
    #         nn.init.constant_(self.W.weight, 0)
    #         nn.init.constant_(self.W.bias, 0)

    #     self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
    #                          kernel_size=1, stride=1, padding=0)

    #     self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
    #                        kernel_size=1, stride=1, padding=0)

    #     if sub_sample:
    #         self.g = nn.Sequential(self.g, max_pool_layer)
    #         self.phi = nn.Sequential(self.phi, max_pool_layer)

    # def forward(self, x):
    #     '''
    #     :param x: (b, c, t, h, w)
    #     :return:
    #     '''

    #     batch_size = x.size(0)

    #     g_x = self.g(x).view(batch_size, self.inter_channels, -1)
    #     g_x = g_x.permute(0, 2, 1)

    #     theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
    #     theta_x = theta_x.permute(0, 2, 1)
    #     phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
    #     f = torch.matmul(theta_x, phi_x)
    #     N = f.size(-1)
    #     f_div_C = f / N

    #     y = torch.matmul(f_div_C, g_x)
    #     y = y.permute(0, 2, 1).contiguous()
    #     y = y.view(batch_size, self.inter_channels, *x.size()[2:])
    #     W_y = self.W(y)
    #     z = W_y + x

    #     return z


# class NONLocalBlock1D(_NonLocalBlockND):
#     def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
#         super(NONLocalBlock1D, self).__init__(in_channels,
#                                               inter_channels=inter_channels,
#                                               dimension=1, sub_sample=sub_sample,
#                                               bn_layer=bn_layer)


# class NONLocalBlock2D(_NonLocalBlockND):
#     def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
#         super(NONLocalBlock2D, self).__init__(in_channels,
#                                               inter_channels=inter_channels,
#                                               dimension=2, sub_sample=sub_sample,
#                                               bn_layer=bn_layer)


# class NONLocalBlock3D(_NonLocalBlockND):
#     def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
#         super(NONLocalBlock3D, self).__init__(in_channels,
#                                               inter_channels=inter_channels,
#                                               dimension=3, sub_sample=sub_sample,
#                                               bn_layer=bn_layer)



# class CGAFusion(nn.Module):
#     def __init__(self, dim, reduction=8):
#         super(CGAFusion, self).__init__()
#         self.sa = SpatialAttention()
#         self.ca = ChannelAttention(dim, reduction)
#         self.pa = PixelAttention(dim)
#         self.ga = NONLocalBlock2D(dim)

#         self.conv = nn.Conv2d(dim, dim, 1, bias=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, y):
#         initial = x + y
#         cattn = self.ca(initial)
#         sattn = self.sa(initial)
#         gattn = self.ga(initial)
#         pattn1 = sattn + cattn + gattn
#         pattn2 = self.sigmoid(self.pa(initial, pattn1))
#         result = initial + pattn2 * x + (1 - pattn2) * y
#         result = self.conv(result)
#         return result



# class FourierUnit(nn.Module):
#     def __init__(self, embed_dim, fft_norm='ortho'):
#         # bn_layer not used
#         super(FourierUnit, self).__init__()
#         self.conv_layer = torch.nn.Conv2d(embed_dim * 2, embed_dim * 2, 1, 1, 0)
#         self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

#         self.fft_norm = fft_norm

#     def forward(self, x):
#         batch = x.shape[0]

#         r_size = x.size()
#         # (batch, c, h, w/2+1, 2)
#         fft_dim = (-2, -1)
#         ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
#         ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
#         ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
#         ffted = ffted.view((batch, -1,) + ffted.size()[3:])

#         ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
#         ffted = self.relu(ffted)

#         ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4,
#                                                                        2).contiguous()  # (batch,c, t, h, w/2+1, 2)
#         ffted = torch.complex(ffted[..., 0], ffted[..., 1])

#         ifft_shape_slice = x.shape[-2:]
#         output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

#         return output

#

class Fusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(Fusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PA(nf=dim)
        # self.ga = NONLocalBlock2D(dim)

        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        yglobal = x+y #yfine+ycoarse
        ylocal = x #yfine
        # print(x.shape,y.shape,initial.shape)
        cattn = self.ca(yglobal)
        # print(cattn.shape)
        # visualize_channel_weights(cattn.clone().cpu())
        # input()
        sattn = self.sa(yglobal)
        # print("sattn")
        
        # visualize_feature_map(sattn.clone().cpu())
        

        gattn = self.pa(ylocal) #yfine
        # print("gattn")
        # visualize_feature_map(gattn.clone().cpu())

            
        weight = self.sigmoid(cattn + sattn)

        # result = gattn * weight + (y)*(1-weight)
        result = gattn * weight + (y)*(1-weight)
        # visualize_feature_map(result.clone().cpu())
        result = self.conv(result)
        return result

# class Fusion(nn.Module):
#     '''
#     多特征融合 iAFF
#     '''
#
#     def __init__(self, channels=64, r=4):
#         super(Fusion, self).__init__()
#         inter_channels = int(channels // r)
#
#         # 本地注意力
#         self.local_att = nn.Sequential(
#             nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels),
#         )
#
#         # 全局注意力
#         self.global_att = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels),
#         )
#
#         # 第二次本地注意力
#         self.local_att2 = nn.Sequential(
#             nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels),
#         )
#         # 第二次全局注意力
#         self.global_att2 = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels),
#         )
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x, residual):
#         xa = x + residual
#         xl = self.local_att(xa)
#         xg = self.global_att(xa)
#         xlg = xl + xg
#         wei = self.sigmoid(xlg)
#         xi = x * wei + residual * (1 - wei)
#
#         xl2 = self.local_att2(xi)
#         xg2 = self.global_att(xi)
#         xlg2 = xl2 + xg2
#         wei2 = self.sigmoid(xlg2)
#         xo = x * wei2 + residual * (1 - wei2)
#         return xo