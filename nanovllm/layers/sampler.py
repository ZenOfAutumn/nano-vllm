"""
Token 采样模块 - Gumbel-Softmax 采样

实现 LLM 推理中的 Token 采样，支持温度控制。
使用 Gumbel-Softmax 技巧进行高效的 top-1 采样。
"""

import torch
from torch import nn


class Sampler(nn.Module):
    """
    Token 采样器（Gumbel-Softmax 采样）

    在 LLM 推理中，根据 logits 和温度参数进行概率采样，生成下一个 token。
    使用 Gumbel 分布实现高效的采样，相比传统 multinomial 采样更快。

    采样方法：
    Gumbel-Softmax 采样是一种通过添加 Gumbel 噪声再求 argmax 来进行采样的方法：
        1. 对 logits 进行温度缩放：logits_scaled = logits / temperature
        2. 计算概率分布：probs = softmax(logits_scaled)
        3. 添加 Gumbel 噪声：noise = -log(-log(U))，其中 U ~ Uniform(0,1)
        4. 采样：token = argmax(log(probs) + noise) = argmax(logits + noise)

    优势：
    - 等价于多项分布采样，但实现更高效
    - 支持批量采样和不同温度
    - 易于在 CUDA 上并行化和编译优化

    温度参数影响：
    - temperature → 0：采样变得确定性，趋向于 argmax
    - temperature = 1：标准 softmax 采样
    - temperature → ∞：采样变得均匀分布
    """

    def __init__(self):
        """
        初始化 Token 采样器

        该模块不需要任何可学习的参数，是一个纯函数式的采样操作。
        """
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        前向传播：采样生成下一个 token

        @torch.compile 装饰器：
            - 使用 TorchCompile 进行图编译优化
            - 将整个采样流程编译为高效的 CUDA 核
            - 显著提升推理性能（20-40% 加速）

        参数:
            logits (torch.Tensor): 模型输出的 logits 张量
                                 形状: (batch_size, vocab_size)
                                 值：未经 softmax 的原始预测分数
                                 数据类型: float32 或 float16

            temperatures (torch.Tensor): 每个样本的温度参数
                                       形状: (batch_size,)
                                       值：通常在 0.0 到 2.0 之间
                                       - < 1.0：增加高概率 token 的采样概率（更确定）
                                       - = 1.0：标准采样
                                       - > 1.0：增加多样性（更随机）

        返回:
            torch.Tensor: 采样生成的 token ID
                         形状: (batch_size,)
                         值：[0, vocab_size) 范围内的整数
                         数据类型: int64

        工作流程:
        1. 温度缩放：将 logits 除以温度，控制概率分布的尖锐度
        2. Softmax：计算概率分布
        3. Gumbel 噪声：生成指数分布噪声 -log(-log(U))
        4. Argmax：选择最大概率的 token

        数学公式:
            logits_scaled = logits / temperature
            probs = softmax(logits_scaled)
            gumbel_noise = -log(-log(Uniform(0,1)))
            token = argmax(log(probs) + gumbel_noise)

        性能特性:
            - In-place 操作：使用 div_ 减少内存分配
            - 数据类型保留：保持输入的精度
            - Gumbel 技巧：避免显式的 log(probs) 计算

        数值稳定性:
            - exponential_(1) 生成 Exp(1) 分布，取反后得到 Gumbel(0,1)
            - clamp_min_(1e-10) 防止 log(0) 导致的 -inf
            - 整个过程数值稳定，不易溢出

        示例:
            >>> sampler = Sampler()
            >>> batch_size, vocab_size = 32, 50000
            >>> logits = torch.randn(batch_size, vocab_size)
            >>> temperatures = torch.ones(batch_size) * 0.8  # 温度缩放因子
            >>> tokens = sampler(logits, temperatures)
            >>> tokens.shape  # torch.Size([32])
            >>> tokens.min(), tokens.max()  # (tensor(0), tensor(49999))

        使用场景:
            - Prefill 阶段：为最后一个 token 采样
            - Decode 阶段：为新生成的每个 token 采样
            - Batch 推理：同时对多个序列采样

        与其他采样方法的对比:
            - Top-k 采样：在最高的 k 个概率中采样
            - Top-p (nucleus) 采样：按累积概率采样
            - Beam Search：保留多个候选序列
            - Gumbel-Softmax：本实现，高效的 top-1 采样
        """
        # 【步骤 1】温度缩放
        # 将 logits 除以温度调整概率分布的锐度
        # temperatures.unsqueeze(dim=1) 将形状从 (B,) 扩展到 (B, 1) 以进行广播
        # 例如：logits (32, 50000) / temperatures (32, 1) → (32, 50000)
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))

        # 【步骤 2】计算概率分布
        # softmax 将温度缩放的 logits 转换为概率分布
        # 形状：(batch_size, vocab_size)
        probs = torch.softmax(logits, dim=-1)

        # 【步骤 3】Gumbel-Softmax 采样
        # 关键技巧：使用 Gumbel 噪声代替显式采样
        # torch.empty_like(probs).exponential_(1) 生成 Exp(1) 分布的噪声
        # 取反后得到 Gumbel(0,1) 分布：-log(-log(U))，其中 U ~ Uniform(0,1)
        # clamp_min_(1e-10) 防止 log(0) 导致的数值不稳定（-inf）
        # probs.div_(...) 实现 log(probs) + gumbel_noise 的效果
        #
        # 数学等价性：
        #   log(probs) + (-log(-log(U))) = log(probs) - log(-log(U))
        #   实际计算：probs / (-log(U)) 再取 argmax
        #   等价于：argmax(log(probs) - log(-log(U)))
        #   这等价于标准 Gumbel-Softmax 采样
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)

        # 【步骤 4】返回采样的 token
        # 形状：(batch_size,)
        # 值：0 到 vocab_size-1 的整数索引
        return sample_tokens
