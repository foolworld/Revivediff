"""
Retinex-augmented LQGT Dataset for ReviveDiff (方案B).

方案B：Retinex 分解（反射率 R + 光照 I）作为额外的条件输入通道，拼接到原始 LQ 上。
  - 数据集输出：LQ (9ch) = [原始LQ(3ch) | R(3ch) | I(3ch)]
  - 网络 intro 层：Conv2d(12, 64) — 噪声状态(3ch) + LQ(9ch) = 12ch
  - SDE 均值 μ：仍然使用原始 LQ 的前 3 通道（不变）
  - SDE 动力学：完全不变

相比方案A的优势：
  - μ 不被 Retinex 伪影污染，SDE 过程完全不变
  - R/I 只是辅助信息，模型可自行学习是否利用

使用方法：
  1. 确保 revivediff.yml 中 train 和 val 都有 retinex_method: msrcr
  2. 确保 network_G.setting 中有 cond_channels: 9
  3. 训练时仍然运行 python train.py -opt options/train/revivediff.yml
  4. 恢复原始版本：删掉 retinex_method，设置 cond_channels: 3（或不设置，默认3）

Retinex 算法说明：
  - SSR  (Single Scale Retinex)：单尺度，速度快
  - MSR  (Multi-Scale Retinex)：多尺度，更鲁棒
  - MSRCR (MSR with Color Restoration)：带色彩恢复，效果最好（默认）

参考文献：
  - D.J. Jobson et al., "A Multiscale Retinex for Bridging the Gap Between Color Images
    and the Human Observation of Scenes", IEEE TIP, 1997.
"""

import os
import random
import sys

import cv2
try:
    import lmdb
except ImportError:
    lmdb = None  # lmdb 仅在使用 data_type: lmdb 时需要
import numpy as np
import torch
import torch.utils.data as data

try:
    sys.path.append("..")
    import data.util as util
except ImportError:
    pass


# ==============================================================================
#                          Retinex 算法实现
# ==============================================================================

def _single_scale_retinex(img, sigma):
    """
    单尺度 Retinex (SSR)
    
    Args:
        img:  np.float32, HWC, [0,1], BGR
        sigma: 高斯核标准差
    
    Returns:
        retinex: np.float32, HWC, 对数域反射率
    """
    # 高斯模糊 → 估计光照分量（在 [0,1] 线性域）
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    # 对数域做减法 → 反射率
    eps = 1e-6
    retinex = np.log(img + eps) - np.log(blur + eps)
    return retinex


def _multi_scale_retinex(img, scales=(15, 80, 250), weights=None):
    """
    多尺度 Retinex (MSR)
    
    Args:
        img:     np.float32, HWC, [0,1], BGR
        scales:  高斯核尺度列表
        weights: 各尺度的权重，默认等权
    
    Returns:
        retinex: np.float32, HWC, 对数域反射率
    """
    if weights is None:
        weights = [1.0 / len(scales)] * len(scales)
    
    retinex = np.zeros_like(img)
    for sigma, w in zip(scales, weights):
        retinex += w * _single_scale_retinex(img, sigma)
    
    return retinex


def _msrcr(img, scales=(15, 80, 250), weights=None,
           alpha=125.0, beta=46.0, 
           G=192.0, b=-30.0):
    """
    带色彩恢复的多尺度 Retinex (MSRCR)
    
    这是效果最好的 Retinex 变体，在 MSR 基础上加入了色彩恢复系数，
    避免增强后图像变灰。
    
    Args:
        img:     np.float32, HWC, [0,1], BGR
        scales:  高斯核尺度列表
        weights: 各尺度的权重
        alpha:   色彩恢复强度（控制非线性的程度）
        beta:    色彩恢复增益
        G:       最终增益 (gain)
        b:       最终偏移 (offset)
    
    Returns:
        enhanced: np.float32, HWC, [0,1], 增强后的 BGR 图像
    """
    # Step 1: MSR（对数域）
    msr = _multi_scale_retinex(img, scales, weights)  # HWC
    
    # Step 2: 色彩恢复系数
    # C_i(x,y) = beta * log(alpha * I_i(x,y) / sum_j I_j(x,y))
    eps = 1e-6
    img_sum = np.sum(img, axis=2, keepdims=True)  # HWC, sum over channels
    color_restoration = beta * np.log(alpha * img / (img_sum + eps) + eps)
    
    # Step 3: MSRCR = color_restoration * MSR（逐通道乘法）
    msrcr = color_restoration * msr  # HWC
    
    # Step 4: 增益/偏移 映射到 [0,1]
    enhanced = G * msrcr + b
    enhanced = np.clip(enhanced, 0, 1)
    
    return enhanced.astype(np.float32)


def _ssr_enhance(img, sigma=80, G=192.0, b=-30.0):
    """
    单尺度 Retinex 增强（简化版，速度快）
    
    Args:
        img:   np.float32, HWC, [0,1], BGR
        sigma: 高斯核标准差
        G:     增益
        b:     偏移
    
    Returns:
        enhanced: np.float32, HWC, [0,1]
    """
    retinex = _single_scale_retinex(img, sigma)
    enhanced = G * retinex + b
    enhanced = np.clip(enhanced, 0, 1)
    return enhanced.astype(np.float32)


def _msr_enhance(img, scales=(15, 80, 250), weights=None, G=192.0, b=-30.0):
    """
    多尺度 Retinex 增强（不带色彩恢复，但速度快于 MSRCR）
    
    Args:
        img:   np.float32, HWC, [0,1], BGR
        scales: 高斯核尺度列表
        weights: 各尺度的权重
        G:     增益
        b:     偏移
    
    Returns:
        enhanced: np.float32, HWC, [0,1]
    """
    retinex = _multi_scale_retinex(img, scales, weights)
    enhanced = G * retinex + b
    enhanced = np.clip(enhanced, 0, 1)
    return enhanced.astype(np.float32)


def apply_retinex(img, method='msrcr', **kwargs):
    """
    对 BGR 图像应用 Retinex 增强
    
    Args:
        img:    np.float32, HWC, [0,1], BGR
        method: 'ssr' | 'msr' | 'msrcr'
        **kwargs: 传递给具体方法的参数
    
    Returns:
        enhanced: np.float32, HWC, [0,1], BGR
    """
    if method == 'ssr':
        return _ssr_enhance(img, **kwargs)
    elif method == 'msr':
        return _msr_enhance(img, **kwargs)
    elif method == 'msrcr':
        return _msrcr(img, **kwargs)
    else:
        raise ValueError(f"Unknown Retinex method: {method}. Choose from 'ssr', 'msr', 'msrcr'.")


def apply_retinex_decomposition(img, method='msrcr'):
    """
    将图像分解为反射率 R 和光照 I（方案B：作为额外输入通道）。
    
    Retinex 理论: S(x,y) = R(x,y) * L(x,y)
    对数域: log(S) = log(R) + log(L) → log(R) = log(S) - log(L)
    
    Args:
        img:    np.float32, HWC, [0,1], BGR
        method: 'ssr' | 'msr' | 'msrcr'
    
    Returns:
        R: np.float32, HWC, [0,1] — 反射率（高频细节，接近 GT 纹理）
        I: np.float32, HWC, [0,1] — 光照分量（平滑亮度，辅助理解全局光照）
    """
    eps = 1e-6
    img_f = img.astype(np.float32)

    if method == 'ssr':
        sigma = 80
        I = cv2.GaussianBlur(img_f, (0, 0), sigma)
        log_R = np.log(img_f + eps) - np.log(I + eps)
    elif method in ('msr', 'msrcr'):
        scales = (15, 80, 250)
        log_R = np.zeros_like(img_f)
        for sigma in scales:
            I_s = cv2.GaussianBlur(img_f, (0, 0), sigma)
            log_R += (np.log(img_f + eps) - np.log(I_s + eps)) / len(scales)
        # 取中间尺度估计光照分量
        I = cv2.GaussianBlur(img_f, (0, 0), scales[1])
    else:
        raise ValueError(f"Unknown Retinex method: {method}")

    R = np.exp(log_R)
    R = np.clip(R, 0, 1).astype(np.float32)
    I = np.clip(I, 0, 1).astype(np.float32)
    return R, I


# ==============================================================================
#                          Retinex-enhanced Dataset
# ==============================================================================

class LQGTDatasetRetinex(data.Dataset):
    """
    在 LQGTDataset 基础上，对 LQ 图像应用 Retinex 增强预处理。
    
    Retinex 增强后的 LQ 图像更接近 GT 的图像质量，作为 SDE 的均值回归目标 μ，
    可以缩短扩散距离，使模型更容易学习。
    
    额外支持的配置项（在 dataset_opt 中设置）：
        retinex_method:   Retinex 方法，可选 'ssr' / 'msr' / 'msrcr'（默认 'msrcr'）
        retinex_on_train: 是否只在训练时应用（默认 True，测试时用原始 LQ）
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.LR_paths, self.GT_paths = None, None
        self.LR_env, self.GT_env = None, None
        self.LR_size, self.GT_size = opt["LR_size"], opt["GT_size"]

        # ---- Retinex 配置 ----
        self.retinex_method = opt.get("retinex_method", "msrcr")
        self.retinex_on_train = opt.get("retinex_on_train", True)

        # read image list
        if opt["data_type"] == "lmdb":
            self.LR_paths, self.LR_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_LQ"]
            )
            self.GT_paths, self.GT_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_GT"]
            )
        elif opt["data_type"] == "img":
            self.LR_paths = util.get_image_paths(opt["data_type"], opt["dataroot_LQ"])
            self.GT_paths = util.get_image_paths(opt["data_type"], opt["dataroot_GT"])
        else:
            print("Error: data_type is not matched in Dataset")
        
        assert self.GT_paths, "Error: GT paths are empty."
        if self.LR_paths and self.GT_paths:
            assert len(self.LR_paths) == len(self.GT_paths), \
                "GT and LR datasets have different number of images - {}, {}.".format(
                    len(self.LR_paths), len(self.GT_paths)
                )
        self.random_scale_list = [1]

    def _init_lmdb(self):
        self.GT_env = lmdb.open(
            self.opt["dataroot_GT"], readonly=True, lock=False,
            readahead=False, meminit=False,
        )
        self.LR_env = lmdb.open(
            self.opt["dataroot_LQ"], readonly=True, lock=False,
            readahead=False, meminit=False,
        )

    def __getitem__(self, index):
        if self.opt["data_type"] == "lmdb":
            if (self.GT_env is None) or (self.LR_env is None):
                self._init_lmdb()

        GT_path, LR_path = None, None
        scale = self.opt["scale"] if self.opt["scale"] else 1
        GT_size = self.opt["GT_size"]
        LR_size = self.opt["LR_size"]

        # ---- 读 GT ----
        GT_path = self.GT_paths[index]
        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.GT_sizes[index].split("_")]
        else:
            resolution = None
        img_GT = util.read_img(self.GT_env, GT_path, resolution)

        if self.opt["phase"] != "train":
            img_GT = util.modcrop(img_GT, scale)

        # ---- 读 LQ ----
        if self.LR_paths:
            LR_path = self.LR_paths[index]
            if self.opt["data_type"] == "lmdb":
                resolution = [int(s) for s in self.LR_sizes[index].split("_")]
            else:
                resolution = None
            img_LR = util.read_img(self.LR_env, LR_path, resolution)
        else:
            if self.opt["phase"] == "train":
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_GT.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(np.copy(img_GT), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_GT.shape
            img_LR = util.imresize(img_GT, 1 / scale, True)
            if img_LR.ndim == 2:
                img_LR = np.expand_dims(img_LR, axis=2)

        # ============================================================
        #  方案B：Retinex 分解 → 反射率 R + 光照 I 作为额外输入通道
        #  先转 RGB 再做分解，保证所有通道统一为 RGB 顺序
        # ============================================================
        if self.retinex_method is not None:
            img_LR_rgb = img_LR[:, :, [2, 1, 0]]  # BGR → RGB
            R, I_ch = apply_retinex_decomposition(img_LR_rgb, method=self.retinex_method)
            img_LR = np.concatenate([img_LR_rgb, R, I_ch], axis=2)  # HWC: 3→9，全 RGB
        # ============================================================

        # ---- 训练时的裁剪和增强 ----
        if self.opt["phase"] == "train":
            H, W, C = img_LR.shape
            assert LR_size == GT_size // scale, "GT size does not match LR size"

            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            img_LR, img_GT = util.augment(
                [img_LR, img_GT], self.opt["use_flip"], self.opt["use_rot"],
                self.opt["mode"], self.opt["use_swap"],
            )
        elif LR_size is not None:
            H, W, C = img_LR.shape
            assert LR_size == GT_size // scale, "GT size does not match LR size"
            if LR_size < H and LR_size < W:
                rnd_h = H // 2 - LR_size // 2
                rnd_w = W // 2 - LR_size // 2
                img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
                rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

        # ---- 颜色空间转换 ----
        if self.opt["color"]:
            H, W, C = img_LR.shape
            img_LR = util.channel_convert(C, self.opt["color"], [img_LR])[0]
            img_GT = util.channel_convert(img_GT.shape[2], self.opt["color"], [img_GT])[0]

        # ---- BGR → RGB, HWC → CHW, numpy → tensor ----
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
        # 9ch: Retinex 分解时已转为 RGB，无需再交换
        # 3ch: BGR → RGB
        if img_LR.shape[2] == 3:
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))
        ).float()
        img_LR = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))
        ).float()

        if LR_path is None:
            LR_path = GT_path

        return {"LQ": img_LR, "GT": img_GT, "LQ_path": LR_path, "GT_path": GT_path}

    def __len__(self):
        return len(self.GT_paths)


# ==============================================================================
#  快速测试：运行此文件可查看 Retinex 对输入图片的效果
# ==============================================================================

if __name__ == "__main__":
    """
    测试 Retinex 预处理效果：
      python data/LQGT_retinex_dataset.py <LQ图片路径>
    
    会在同目录生成 *_retinex_ssr.png, *_retinex_msr.png, *_retinex_msrcr.png
    三张不同方法的增强结果，方便对比。
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default=None, help="要测试的 LQ 图片路径")
    args = parser.parse_args()

    if args.image is None:
        print("Usage: python data/LQGT_retinex_dataset.py <image_path>")
        print("示例: python data/LQGT_retinex_dataset.py datasets/rain100L/train/input/001.png")
        sys.exit(0)
    
    img_path = args.image
    base = os.path.splitext(img_path)[0]
    
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: 无法读取图片 {img_path}")
        sys.exit(1)
    
    img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    
    print(f"输入图片: {img_path}  shape={img.shape}  range=[{img.min():.3f}, {img.max():.3f}]")
    
    for method in ['ssr', 'msr', 'msrcr']:
        result = apply_retinex(img, method=method)
        result_uint8 = (result * 255).round().clip(0, 255).astype(np.uint8)
        save_path = f"{base}_retinex_{method}.png"
        cv2.imwrite(save_path, result_uint8)
        print(f"[{method}] 已保存: {save_path}  range=[{result.min():.3f}, {result.max():.3f}]")
    
    print("\n完成！请查看生成的 *_retinex_*.png 文件对比不同 Retinex 方法的效果。")
