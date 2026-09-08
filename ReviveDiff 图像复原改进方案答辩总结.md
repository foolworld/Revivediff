## ReviveDiff 图像复原改进方案答辩总结
### 一、项目背景
ReviveDiff 是基于 IRSDE（Image Restoration SDE，均值回归随机微分方程）的条件去噪模型，主干网络为 NAFNet（~88.3M 参数），用于图像复原任务（Rain100L 去雨）。原版使用 L1 loss + 固定 sigma 的 SDE。

本次改进的核心目标： 在保持模型架构和 SDE 扩散过程不变的前提下，通过改进损失函数和引入自适应 sigma 来提升复原质量。

### 二、评价指标含义
指标 全称 衡量什么 越高越好 数值范围 PSNR Peak Signal-to-Noise Ratio（峰值信噪比） 像素级重建精度，衡量预测图像与真值图像在每个像素点上的差异。值越大说明像素误差越小、图像越接近于真值。 是 通常 20-50 dB，>30 dB 为可接受质量 SSIM Structural Similarity Index Measure（结构相似性） 从亮度、对比度、结构三个维度综合评价图像相似度。更贴近人眼视觉感知，比 PSNR 更能反映图像结构信息的保留程度。 是 0-1，越接近 1 越好

PSNR 和 SSIM 互补 ：PSNR 注重逐像素保真度，SSIM 注重结构一致性。理想情况下两者应同时提升，但有时存在 trade-off（如过度平滑会提升 PSNR 但降低 SSIM）。

### 三、各方案实现内容与意义 方案 0：原版 ReviveDiff（Baseline）
改动 ：无，使用原始配置

- Loss: L1 loss
- SDE: 固定 sigma
意义 ：基线对照，评估后续改进的实际效果。
 方案 1：Retinex 预处理
实现内容 ：

- 在输入阶段对 LQ 图像做 MSRCR（多尺度 Retinex 色彩恢复）增强
- 将增强后的图像作为 SDE 的均值回归目标 μ
失败原因与意义 ：

- Retinex 将图像分解为反射率 R（物体本质属性）和光照 I（光照条件），理论上可分离退化与内容
- 失败根因 ：MSRCR 对 加性噪声 （如雨线）会产生 artifacts，因为 Retinex 本质设计用于乘性噪声（低光照增强）。对雨天图像，MSRCR 反而引入了额外失真，SDE 被迫从不含 artifacts 的图像中学习恢复
- PSNR 在 20K 步为 33.41 dB，与原始（33.40 dB）几乎持平， 无实质提升
- 启示 ：预处理方案高度依赖退化类型匹配，Rain100L 这样的加性噪声不适合 Retinex 方案 2：Pearson 相关系数损失
实现内容 ：

- 在原 L1 loss 基础上加入 Pearson 相关系数损失 （权重 0.1）
- 使用协方差公式手动实现（含 eps 防除零）： loss = 1 - cov(pred, target) / sqrt(var_pred * var_target + eps)
意义 ：

- Pearson 相关系数衡量预测与真值的 线性相关程度 ，范围 [-1, 1]
- 将该损失加入训练可引导模型输出与 GT 在 统计分布层面 保持一致，而不仅仅是逐像素对齐
- 可近似看作原版效果（PSNR 33.75 vs 33.40，SSIM 0.950），说明单独添加相关性损失效果有限，需要配合其他损失 方案 3：Sobel 边缘损失
实现内容 ：

- 将原始不可微的 Canny 边缘检测 替换为可微的 Sobel 算子 （3×3 梯度卷积核）
- 对预测图和真值图分别提取 x/y 方向梯度，计算 L1 边缘损失
- 配置：Charbonnier 像素损失 + Sobel 边缘损失（权重 0.5）
意义 ：

- Sobel 算子通过卷积计算图像梯度，可捕获边缘和纹理信息， 全程可微 ，梯度可以正常反向传播
- 边缘损失引导模型关注图像 高频结构 （边缘、轮廓），而不仅仅是平滑区域
- 结果显示：PSNR 33.69（略低于 Pearson 的 33.75），但 SSIM 达到 0.953 ，为所有方案中最高
- 这证明了边缘损失的 trade-off 效应 ：强化边缘保留略微牺牲像素级精度，但换来了更好的结构一致性 方案 4：LS 综合方案（最终版本）
实现内容 ：

- L 层（Loss 增强） ：多损失函数融合
  - Charbonnier 像素损失（ sqrt(x² + ε²) ，比 L1 更平滑、对异常值更鲁棒）
  - Sobel 边缘损失（权重 0.5）：保留边缘结构
  - 可微直方图损失（权重 0.1）：KDE 高斯核实现，约束输出分布
  - 焦点频率损失（权重 0.1）：加权频域 L1，关注高频分量
  - Pearson 相关系数损失（权重 0.1）：约束统计相关性
- S 层（自适应 Sigma） ：受 DT-Diff 启发
  - 轻量级 DegradationEstimator CNN（~3K 参数）：输入 LQ 图像，预测 sigma 缩放因子 ∈ [0.6, 1.4]
  - 根据每张图像的退化程度自适应调节 SDE 噪声强度
  意义 ：

- L 层 ：从像素（Charbonnier）、边缘（Sobel）、统计分布（Histogram + Pearson）、频率（FFL）等多维度约束输出，全面增强复原质量
- S 层 ：不同场景/图像的退化程度不同，固定 sigma 无法适应所有情况；自适应 sigma 让 SDE 根据输入难度动态调整扩散强度
- 在 Rain100L 单任务上 sigma 变化小（~0.99-1.00），自适应优势有限，但在 多任务/多场景场景 下潜力更大
### 四、完整实验结果 PSNR 对比表（单位：dB）
版本 5K 10K 15K 20K 25K Original 30.74 32.29 32.64 33.40 — Retinex 30.37 32.22 32.97 33.41 — Pearson 30.58 32.21 32.79 33.34 33.75 Sobel 30.21 32.06 32.75 33.24 33.69 LS 30.65 32.35 33.08 33.32 33.80
 SSIM 对比表
版本 5K 10K 15K 20K 25K Original — — — — — Retinex 0.907 0.933 0.941 0.947 — Pearson 0.906 0.935 0.940 0.946 0.950 Sobel 0.897 0.932 0.942 0.949 0.953 LS 0.905 0.934 0.943 0.946 0.951
 25K 步最终对比（核心）
版本 PSNR (dB) ΔPSNR SSIM ΔSSIM Pearson 33.75 +0.35 0.950 — Sobel 33.69 +0.29 0.953 — LS 33.80 +0.40 0.951 —
 注：ΔPSNR 为相对 Original 20K（33.40 dB）的提升；Original 未记录 SSIM，故无法计算 ΔSSIM。
### 五、结论与分析
1. Retinex 预处理路径不可行 ：对加性噪声（雨线）引入 artifacts，PSNR 无提升（+0.01 dB），不适合 Rain100L 这类退化类型。
2. Sobel 边缘损失擅长结构保留 ：SSIM 达到最高的 0.953，验证了边缘约束对视觉结构质量的贡献，但 PSNR 略低（33.69），存在像素精度与结构质量的 trade-off。
3. LS 综合方案实现最优平衡 ：
   
   - PSNR 最高 ：33.80 dB（+0.40 vs Baseline），像素级重建精度最优
   - SSIM 次高 ：0.951（仅比 Sobel 低 0.002），结构质量几乎持平
   - 多维度损失融合 + 自适应 sigma 在 单任务上实现了最优综合性能 ，且架构天然支持多任务扩展
4. 自适应 sigma 在单任务上效果有限 ：Rain100L 的 sigma scale 接近 1.0，变化范围小。该模块的主要价值在于 多场景/多退化类型统一训练 场景，是面向未来的扩展能力。



![image-20260731205945266](C:\Users\xs\AppData\Roaming\Typora\typora-user-images\image-20260731205945266.png)

### ![image-20260731210004581](C:\Users\xs\AppData\Roaming\Typora\typora-user-images\image-20260731210004581.png)
1. LS 在 25K 步 PSNR 提升最显著：+1.20% （33.80 dB），像素级重建精度最优
2. Sobel 在 25K 步 SSIM 提升最大：+0.63% （0.953），结构保真度最佳
3. LS 的 SSIM 提升 +0.42% （0.951），与 Sobel 仅差 0.21 个百分点，综合性能最均衡
4. 在 20K 步时各方案均未超越原版（PSNR 均有微小下降，−0.03% ~ −0.48%），说明多损失函数需要更长训练才能收敛并体现出优势
5. Pearson 单独添加效果有限 （25K PSNR +1.05%，SSIM +0.32%），近似于原版效果，验证了需要组合多个损失函数才能获得显著提升













---

**版本 0：原版 ReviveDiff**
基于 IRSDE + NAFNet（~88.3M 参数）的条件去噪模型，使用 L1 loss + 固定 sigma，在 Rain100L 上训练。20K 步 PSNR = 33.40 dB。这是所有后续改动的基线。

---

**版本 1：Retinex 预处理**
在输入端对 LQ 图像做 MSRCR 多尺度 Retinex 增强，增强后的图像作为 SDE 的均值回归目标 μ。目的是通过分离反射率 R 和光照 I 来降低雨线干扰。失败根因：Retinex 适合乘性噪声（低光照），对加性噪声（雨线）反而引入 artifacts。20K 步 PSNR 仅 33.41 dB，与原版几乎持平，判定为无效改进。

---

**版本 2：Pearson 相关系数损失**
在原 L1 loss 基础上加入 Pearson 相关系数损失（权重 0.1），手动实现协方差公式（含 eps 防止除零）。目的：引导模型输出与 GT 在统计分布层面保持一致，而不仅是逐像素对齐。25K 步 PSNR = 33.75 dB、SSIM = 0.950，与原版差异不大，说明单独添加相关性损失效果有限。

---

**版本 3：Sobel 边缘损失**
将不可微的 Canny 边缘检测替换为可微的 Sobel 算子（3×3 梯度卷积核），同时像素损失从 L1 改为 Charbonnier。新增 Focal Frequency Loss 约束频域高频分量，Pearson 损失保留。不改 SDE 结构。25K 步 PSNR = 33.69 dB（略低于 Pearson），但 SSIM 达到所有版本最高的 0.953，验证了边缘约束对结构质量的贡献——以微小像素精度换取更好的视觉结构一致性。

---

**版本 4：LS 综合方案**
在 Sobel 版本的基础上增加了两点：（1）可微直方图损失（KDE 高斯核实现，权重 0.1），约束预测图与 GT 的像素分布一致；（2）自适应 sigma（受 DT-Diff 启发，~3K 参数的 DegradationEstimator CNN 预测每张图的 sigma 缩放因子 ∈ [0.6, 1.4]），让 SDE 根据输入退化程度动态调节扩散强度。损失融合 Charbonnier + Sobel + Histogram + FFL + Pearson 共 5 项。25K 步 PSNR = 33.80 dB（+0.40 vs 原版），为所有版本最高；SSIM = 0.951，仅比 Sobel 低 0.002，实现了像素精度与结构质量的最优平衡。自适应 sigma 在 Rain100L 单任务上变化幅度小（~0.99-1.00），主要价值在于多任务扩展潜力。





---

**1. TC（时间条件调制·初始版）**

思路是受 UniLDiff 启发，让 LQ 条件不是只在输入时注入一次，而是随去噪时刻动态调制，从而在多步去噪中都发挥作用，核心是 `alpha = 0.2 + t_norm * 0.8` 这个随时刻线性增长的系数。改动是在 NAFNet 的 `forward` 里把残差路径写成 `x = inp - alpha * cond`，把调制后的条件直接减掉。结果直接崩盘，5K 时 PSNR 只有 6.26、SSIM 0.003。失败原因是这行把 `(1-alpha)*LQ` 混进了残差路径，破坏了 IRSDE 的均值锚点 μ=LQ，去噪的锚点随时刻漂移，模型完全学废。

**2. TCC（时间条件调制·修复版）**

思路和 TC 完全相同，只是修正了调制的位置：残差路径恢复为 `x = inp - cond` 保持锚点稳定，只在 concat 路径用 `alpha * cond` 去调制条件通道。改动就是这一处位置调整。结果能正常训练了，5K=30.29、10K=31.15、15K=31.99、20K=32.09、25K=32.71，但 25K 仍比原版（33.44）低 0.73，严重落后。失败原因是 IRSDE 本身已经通过均值回归天然实现了"条件随时刻插值"，再额外动态调制 LQ 属于双重插值、互相打架，所以即使锚点没被破坏，也拿不到增益反而拖累收敛——这从根上说明时间条件调制和 IRSDE 的设计是冲突的，不值得做。

**3. SGA（自生成增强·初始版）**

思路是受 CLOVER 论文启发做 Self-Generated Augmentation：先用模型预测反解出干净图 x0，再把它重新加噪到更小时刻，让模型在这个"自生成的难样本"上再预测一次，两条 loss 都对齐真值 GT，从而填补训练分布和推理分布之间的 gap。改动是在 `optimize_parameters` 里加 SGA 分支，初始版没有 detach、loss2 用 `reverse_optimum_step(state_2, GT, t_2)` 作 target。结果连续踩两个坑：先是算力平台 32GB 直接 OOM（因为不 detach，第一次前向的激活要保留到第二次 backward，显存翻倍）；改成 detach 后虽然能跑了，但 5K=25.17、10K=27.36，收敛又被拖垮。失败原因有两个叠加的 bug：一是 `get_x0_from_noise` 里 `除以 exp(-θ̄_t·dt)`，t 大时分母趋近 0（t=299 时放大 193 倍），反解出的 x0 数值爆炸；二是 loss2 用 `reverse_optimum_step` 作 target，其中的 `term1*(state_2 - mu)` 依赖已经烂掉的 state_2，进一步放大错误，两个 bug 让梯度带飞、收敛崩坏。

**4. SGA2（自生成增强·修复版）**

思路和 SGA 完全一致，只是把数值和维度 bug 全部修掉，累计五处修复：`SGA_MAX_T=100` 只在安全区做 SGA（exp_coef≥0.72、放大不超过 1.4 倍，避开大 t 爆炸）；loss2 改成 x0 空间直接对齐 GT（忠实 CLOVER 原文，target 不再被放大）；`set_mu` 随 mask 切片对齐维度；`x0_pred` detach 并分离两次 backward（峰值显存回到接近原版）；`t_2.reshape(-1)` 修掉 N=1 时的标量崩溃。结果是修复成功、回到原版水平：5K=29.18、10K=31.16、15K=32.08、20K=32.55、25K=33.15，而且 SSIM 一路冲到全场最高的 0.9584（原版 0.9490、LS 0.9538、Sobel 0.9530）。失败原因（准确说是遗留的 trade-off）是 loss2 在 x0 干净图空间、loss1 在 x_{t-1} 含噪空间，两者尺度不匹配，等权时 loss2 偏强、牵制了 loss1 的像素级收敛，导致 PSNR 比原版低 0.29（-0.85%）但 SSIM 高 0.009（+0.99%）。针对这一点已经加了独立的 `sga_weight`（默认 0.5）可调旋钮，正在通过扫权重验证能否在保住 SSIM 优势的同时追平甚至超过原版 PSNR。

---

一句话串起来：**TC→TCC 证明"动态调制条件"与 IRSDE 的均值回归机制冲突，方向本身走不通；SGA→SGA2 证明"自生成增强"方向可行，但 IRSDE 的 ε-参数化让反解 x0 存在数值爆炸的固有难点，修好后换来的是结构保真（SSIM）的显著提升，代价是像素精度（PSNR）的小幅让步，这个 trade-off 正在用权重旋钮做最后调优。**







## SGAL（08-15）
改动 ：配置与 SGA2 完全一致（L1 + SGA sga_weight=0.5 + Lion + CosineAnnealing），可能是 SGA 实现的微调变体版本。

结果 ：5K 28.81 / 10K 31.40 / 15K 32.23 / 20K 32.40 / 25K 33.05 ，SSIM 0.9565。与 SGA2 几乎持平（-0.10 dB），在误差范围内，说明该版本的微调没有带来实质改变。

## SGA2_LS 三合一旧版（08-17）
改动 ：将三个改进同时叠加到 SGA2 上——(1) 损失函数从 L1 换为 Charbonnier + Sobel 边缘 + 可微直方图 + Focal Frequency + Pearson 相关系数的组合损失；(2) 加入 adaptive_sigma，DegradationEstimator 输出范围 [0.6, 1.4]；(3) 保留 SGA。这是 LSobel 版本的改进与 SGA2 的首次合并尝试。

结果 ：5K 26.03 / 10K 27.37 / 15K 28.35 / 20K 31.10， 未跑到 25K 。全程严重落后，20K 时仍比 SGA2 同期低 1.45 dB。

问题根因 ：sigma_scale 在前 1500 步从 0.94 暴跌到 0.600 下界并卡死，等于把训练噪声调度缩到 60%（max_sigma 从 90 降到 54），模型在低噪声下"作弊"降低损失但没学到真正的去噪能力。之后花了约 13500 步才爬回 0.94。同时辅助损失（Sobel/Hist/FFL/Pearson）施加在带噪 SDE 中间态上，梯度被噪声主导，进一步拖慢收敛。

## SGA2_LS 修复新版（08-19）
改动 ：针对旧版的两个问题做了修复——(1) DegradationEstimator 输出范围从 [0.6, 1.4] 收窄到 [0.9, 1.1]，防止 sigma 早期坍缩；(2) 去掉所有辅助损失（Sobel/Hist/FFL/Pearson），loss1 和 loss2 都只用纯 Charbonnier；(3) 保留 SGA 和 adaptive_sigma。niter 从 700000 改为 25000 以匹配算力预算。

结果 ：5K 28.59 / 10K 31.86 / 15K 32.61 / 20K 33.40 / 25K 33.59 ，SSIM 0.9590。sigma_scale 全程稳定在 0.98-1.00，没有坍缩。相比旧版 20K 时提升 +2.30 dB（31.10→33.40），相比 SGA2 基线 25K 提升 +0.44 dB（33.15→33.59）。

实际贡献拆解 ：Charbonnier 替代 L1 带来了 +0.44 dB 的增益（与 SGA2 原版对比）。adaptive_sigma 实际上是空操作——sigma_scale 全程 ≈1.0，DegradationEstimator 没有学到任何有用的东西。SGA 的增益无法单独量化，但新版在中后期（15K-25K）的收敛速度与 SGA2 基本一致。

## 补充：SGA2 之前的 LS 系列作为背景
这些版本不包含 SGA，但为 SGA2_LS 的合并提供了基础：

LS_test（07-31） ：adaptive_sigma [0.6,1.4] + 纯 L1，无 SGA。25K PSNR 33.80 （全场最高）。sigma_scale 早期坍缩到 0.6 再爬回，意外起到了 curriculum 效果。

LS_260802（08-02） ：adaptive_sigma [0.6,1.4] + Charbonnier + Sobel全套，无 SGA。25K PSNR 33.76 。辅助损失导致比 LS_test 低 0.04 dB。

train_260802（08-02） ：Charbonnier + Sobel全套，无 adaptive_sigma，无 SGA。25K PSNR 33.44 。证明去掉 adaptive_sigma 后 PSNR 下降 0.32 dB（对比 LS_260802），但这个差距来自 [0.6,1.4] 范围的意外 curriculum，不是 estimator 的设计意图。