import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import sys
import cv2
import numpy as np

def canny_edge_detector(tensor_image):
    # 将张量转换为NumPy数组
    np_image = tensor_image.detach().cpu().numpy().transpose(1, 2, 0)
    np_image = (np_image * 255).astype(np.uint8)

    # 转换为灰度图
    gray_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    # 应用高斯模糊
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    v = np.median(blurred_image)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    # 应用Canny边缘检测
    canny_edges = cv2.Canny(blurred_image, lower, upper)

    # 将NumPy数组转换回张量
    edge_tensor = torch.from_numpy(canny_edges).to(tensor_image.device).float()
    edge_tensor = edge_tensor.unsqueeze(0)  # 增加一个通道维度
    return edge_tensor

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()

    def forward(self, enhanced_batch, hq_batch):
        batch_loss = 0.0

        # 遍历批次中的每个图像
        for enhanced_tensor, hq_tensor in zip(enhanced_batch, hq_batch):
            # 应用边缘检测
            edge_enhanced = canny_edge_detector(enhanced_tensor)
            edge_hq = canny_edge_detector(hq_tensor)

            # 计算单个图像的损失
            loss = F.l1_loss(edge_enhanced, edge_hq, reduction='none')
            batch_loss += loss

        # 计算批次的平均损失
        return batch_loss / len(enhanced_batch)






def normalize_tensor(tensor):
    # 将张量归一化到 [0, 1] 范围
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

def histogram_loss(enhanced_tensor, hq_tensor, bins=256):
    # 检查输入是否为单个图像，如果是，则增加批次维度
    if len(enhanced_tensor.shape) == 3:
        enhanced_tensor = enhanced_tensor.unsqueeze(0)
    if len(hq_tensor.shape) == 3:
        hq_tensor = hq_tensor.unsqueeze(0)

    # 归一化张量
    enhanced_tensor = normalize_tensor(enhanced_tensor)
    hq_tensor = normalize_tensor(hq_tensor)

    # 初始化损失
    loss = 0.0

    # 对每个通道计算直方图并计算损失
    for channel in range(enhanced_tensor.shape[1]):
        # 计算直方图
        hist_enhanced = torch.histc(enhanced_tensor[:, channel, :, :], bins=bins, min=0, max=1)
        hist_hq = torch.histc(hq_tensor[:, channel, :, :], bins=bins, min=0, max=1)

        # 归一化直方图
        hist_enhanced = hist_enhanced / torch.sum(hist_enhanced)
        hist_hq = hist_hq / torch.sum(hist_hq)

        # 计算当前通道的损失
        loss += F.l1_loss(hist_enhanced, hist_hq, reduction='none')

    # 返回平均损失
    return loss / enhanced_tensor.shape[1]


def rgb_to_lab(tensor):
    # 检查输入是否为单个图像，如果是，则增加批次维度
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)

    # 确保张量在CPU上，并转换为NumPy数组
    np_image = tensor.detach().cpu().numpy()

    # 转换为LAB颜色空间
    lab_images = []
    for img in np_image:
        img = img.transpose(1, 2, 0)  # 转换为HWC格式
        img = (img * 255).astype(np.uint8)
        lab_image = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_channel = lab_image[:, :, 0]  # 提取L通道
        lab_images.append(l_channel)

    # 将列表转换回张量
    l_channel_tensor = torch.tensor(lab_images, dtype=torch.float32).to(tensor.device)
    l_channel_tensor = l_channel_tensor.unsqueeze(1)  # 增加通道维度
    return l_channel_tensor / 255.0  # 归一化


def histogram_light_loss(enhanced_tensor, hq_tensor, bins=256):
    # 检查输入是否为单个图像，如果是，则增加批次维度
    if len(enhanced_tensor.shape) == 3:
        enhanced_tensor = enhanced_tensor.unsqueeze(0)
    if len(hq_tensor.shape) == 3:
        hq_tensor = hq_tensor.unsqueeze(0)

    # 归一化张量
    enhanced_tensor = normalize_tensor(enhanced_tensor)
    hq_tensor = normalize_tensor(hq_tensor)

    # 初始化损失
    loss = 0.0

    # 对每个通道计算直方图并计算损失
    for channel in range(enhanced_tensor.shape[1]):
        # 计算直方图
        hist_enhanced = torch.histc(enhanced_tensor[:, channel, :, :], bins=bins, min=0, max=1)
        hist_hq = torch.histc(hq_tensor[:, channel, :, :], bins=bins, min=0, max=1)

        # 归一化直方图
        hist_enhanced = hist_enhanced / torch.sum(hist_enhanced)
        hist_hq = hist_hq / torch.sum(hist_hq)

        # 计算当前通道的损失
        loss += F.l1_loss(hist_enhanced, hist_hq)

    # 计算亮度通道的直方图损失
    l_enhanced = rgb_to_lab(enhanced_tensor)
    l_hq = rgb_to_lab(hq_tensor)
    hist_l_enhanced = torch.histc(l_enhanced, bins=bins, min=0, max=1)
    hist_l_hq = torch.histc(l_hq, bins=bins, min=0, max=1)
    hist_l_enhanced = hist_l_enhanced / torch.sum(hist_l_enhanced)
    hist_l_hq = hist_l_hq / torch.sum(hist_l_hq)
    loss += F.l1_loss(hist_l_enhanced, hist_l_hq, reduction='none')

    # 返回平均损失
    return loss / (enhanced_tensor.shape[1] + 1)  # 加1是因为加入了亮度通道


def fourier_transform(tensor):
    # 假设tensor是一个形状为(batch_size, channels, height, width)的张量
    # 执行傅里叶变换
    fft = torch.fft.fft2(tensor)
    # 将复数转换为幅度
    amplitude = torch.abs(fft)
    return amplitude

def fourier_loss(enhanced_tensor, original_tensor):
    # 执行傅里叶变换
    fft_enhanced = fourier_transform(enhanced_tensor)
    fft_original = fourier_transform(original_tensor)

    # 计算损失（例如，使用L1损失）
    loss = F.l1_loss(fft_enhanced, fft_original, reduction='none')
    return loss

# def combined_loss(enhanced_tensor, hq_tensor, bins=256, alpha=1.0):
#     # 基本损失计算
#     pixel_loss_val = F.l1_loss(enhanced_tensor, hq_tensor)
#     edge_loss_val = edge_loss(enhanced_tensor, hq_tensor)
#     hist_loss_val = histogram_loss(enhanced_tensor, hq_tensor, bins)
#     fourier_loss_val = fourier_loss(enhanced_tensor, hq_tensor)

#     # 非线性权重调整
#     weight_edge = torch.sigmoid(fourier_loss_val)
#     weight_hist = torch.sigmoid(fourier_loss_val)

#     # 组合损失
#     total_loss = alpha * pixel_loss_val + weight_edge * edge_loss_val + weight_hist * hist_loss_val
#     return total_loss





class MatchingLoss(nn.Module):
    def __init__(self, loss_type='l1', is_weighted=False):
        super().__init__()
        self.is_weighted = is_weighted

        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')
        self.EG = EdgeLoss()

    def forward(self, predict, target, weights=None):

        # loss = self.loss_fn(predict, target, reduction='none')
        # loss = histogram_light_loss(predict, target) + self.EG(predict, target)+ fourier_loss(predict, target)+ loss

        pixel_loss_val = self.loss_fn(predict, target, reduction='none')
        edge_loss_val = self.EG(predict, target)
        # NOTE: This prior loss is configured for the manually chosen training
        # patch size. With bins=256, hist_loss_val has length 256 and is
        # broadcast together with pixel_loss_val; keep GT_size/LR_size aligned
        # with this setting, or adjust the bin count consistently.
        hist_loss_val = histogram_loss(predict, target, 256)
        # fourier_loss_val = fourier_loss(predict, target)

        # 非线性权重调整
        weight_edge = 1
        # weight_hist = torch.sigmoid(fourier_loss_val)
        # print(fourier_loss_val)
        # print(weight_edge,'**********')

        # 组合损失 
        total_loss =  pixel_loss_val  + edge_loss_val + hist_loss_val
        # total_loss =  pixel_loss_val  + edge_loss_val 
        # total_loss =  pixel_loss_val   + hist_loss_val
        # total_loss = pixel_loss_val
        # return total_loss

        loss = einops.reduce(total_loss, 'b ... -> b (...)', 'mean')
        if self.is_weighted and weights is not None:
            loss = weights * loss

        return loss.mean()


