import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier_loss(predict, target, eps=1e-6):
    """Charbonnier 损失，对异常值更鲁棒。"""
    return torch.sqrt((predict - target) ** 2 + eps ** 2).mean()


class SobelEdgeLoss(nn.Module):
    """可微的 Sobel 边缘损失，替换不可导的 Canny 边缘检测。"""
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        kernel_y = kernel_x.transpose(2, 3)
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    def forward(self, predict, target):
        if predict.shape[1] == 3:
            rgb_weight = predict.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            predict_gray = (predict * rgb_weight).sum(dim=1, keepdim=True)
            target_gray = (target * rgb_weight).sum(dim=1, keepdim=True)
        else:
            predict_gray = predict
            target_gray = target

        px = F.conv2d(predict_gray, self.kernel_x, padding=1)
        py = F.conv2d(predict_gray, self.kernel_y, padding=1)
        tx = F.conv2d(target_gray, self.kernel_x, padding=1)
        ty = F.conv2d(target_gray, self.kernel_y, padding=1)

        return F.l1_loss(px, tx) + F.l1_loss(py, ty)


def focal_frequency_loss(predict, target):
    """频域损失，针对高频成分加权，对去雨条纹有效。"""
    f_pred = torch.fft.fft2(predict)
    f_target = torch.fft.fft2(target)
    amp_pred = torch.abs(f_pred)
    amp_target = torch.abs(f_target)
    weight = 1 + torch.abs(amp_target - amp_pred)
    return F.l1_loss(amp_pred * weight, amp_target * weight)


def clamp_tensor(tensor, min_val=0.0, max_val=1.0):
    """限制张量范围到 [min_val, max_val]，不改变原始分布形状。"""
    return tensor.clamp(min=min_val, max=max_val)


def differentiable_histogram(tensor, bins=128, min_val=0, max_val=1):
    """可微软直方图计算，使用 kernel density estimation。"""
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    
    B = tensor.shape[0]
    bin_centers = torch.linspace(min_val, max_val, bins, device=tensor.device)
    bin_centers = bin_centers.view(1, bins, 1)
    bandwidth = (max_val - min_val) / bins
    bandwidth = max(bandwidth, 1e-6)
    chunk_size = 4096
    N = tensor.shape[1]
    hist = torch.zeros(B, bins, device=tensor.device)
    
    for i in range(0, N, chunk_size):
        chunk = tensor[:, i:i+chunk_size].unsqueeze(1)
        chunk_hist = torch.exp(-((chunk - bin_centers) ** 2) / (2 * bandwidth ** 2))
        hist += chunk_hist.sum(dim=-1)
    
    hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-6)
    return hist


def histogram_loss(enhanced_tensor, hq_tensor, bins=128):
    """可微直方图损失，使用软直方图替代 torch.histc。"""
    if len(enhanced_tensor.shape) == 3:
        enhanced_tensor = enhanced_tensor.unsqueeze(0)
    if len(hq_tensor.shape) == 3:
        hq_tensor = hq_tensor.unsqueeze(0)

    # 使用 clamp 代替 normalize_tensor，保留原始亮度信息
    enhanced_tensor = clamp_tensor(enhanced_tensor)
    hq_tensor = clamp_tensor(hq_tensor)

    loss = torch.tensor(0.0, device=enhanced_tensor.device)

    for channel in range(enhanced_tensor.shape[1]):
        enh_flat = enhanced_tensor[:, channel, :, :].reshape(enhanced_tensor.shape[0], -1)
        hq_flat = hq_tensor[:, channel, :, :].reshape(hq_tensor.shape[0], -1)

        hist_enhanced = differentiable_histogram(enh_flat, bins=bins, min_val=0, max_val=1)
        hist_hq = differentiable_histogram(hq_flat, bins=bins, min_val=0, max_val=1)

        hist_enhanced = hist_enhanced / (hist_enhanced.sum(dim=-1, keepdim=True) + 1e-6)
        hist_hq = hist_hq / (hist_hq.sum(dim=-1, keepdim=True) + 1e-6)

        loss = loss + F.l1_loss(hist_enhanced, hist_hq)

    return loss / enhanced_tensor.shape[1]


def pearson_loss_fn(predict, target, eps=1e-6):
    """数值稳定的 Pearson 相关系数损失。"""
    losses = []
    
    for p, t in zip(predict, target):
        p = p.reshape(-1)
        t = t.reshape(-1)
        
        p_centered = p - p.mean()
        t_centered = t - t.mean()
        
        cov = (p_centered * t_centered).mean()
        var_p = p_centered.square().mean()
        var_t = t_centered.square().mean()
        
        corr = cov / torch.sqrt(var_p * var_t + eps)
        losses.append(1.0 - corr)
    
    return torch.stack(losses)


class MatchingLoss(nn.Module):
    """组合损失函数，支持 Charbonnier、Sobel、直方图、频域和 Pearson 损失。"""
    
    def __init__(self, loss_type='charbonnier', is_weighted=False, pearson_weight=0.0,
                 edge_weight=0.5, hist_weight=0.1, freq_weight=0.1):
        super().__init__()
        self.is_weighted = is_weighted
        self.pearson_weight = pearson_weight
        self.edge_weight = edge_weight
        self.hist_weight = hist_weight
        self.freq_weight = freq_weight

        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        elif loss_type == 'charbonnier':
            self.loss_fn = charbonnier_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')
        self.EG = SobelEdgeLoss()

    def forward(self, predict, target, weights=None):
        # 主像素损失
        pixel_loss_val = self.loss_fn(predict, target)
        
        # 边缘损失
        edge_loss_val = self.EG(predict, target)
        
        # 直方图损失
        hist_loss_val = histogram_loss(predict, target, 128)
        
        # 频域损失
        freq_loss_val = focal_frequency_loss(predict, target)

        # 组合损失
        total_loss = (
            pixel_loss_val
            + self.edge_weight * edge_loss_val
            + self.hist_weight * hist_loss_val
            + self.freq_weight * freq_loss_val
        )

        # 如果需要样本加权
        if self.is_weighted and weights is not None:
            loss = total_loss.unsqueeze(0) * weights
            # 确保返回标量，避免 .item() 报错
            loss = loss.sum()
        else:
            loss = total_loss

        # Pearson 损失（可选）
        if self.pearson_weight > 0:
            pearson_val = pearson_loss_fn(predict, target)
            loss = loss + self.pearson_weight * pearson_val.mean()

        # 确保返回标量
        return loss.mean() if loss.dim() > 0 else loss
