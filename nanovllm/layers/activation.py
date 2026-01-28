"""
激活函数模块 - 门控线性单元激活

包含 Gated Linear Unit (GLU) 风格的激活函数实现，
这是现代 LLM 中常用的激活函数组合方式。
"""

import torch
import torch.nn.functional as F
from torch import nn


class SiluAndMul(nn.Module):
    """
    SiLU 和乘法门控激活函数（Swish-Gated Linear Unit）

    实现 SiLU(x) * y 的组合激活，其中 x 和 y 是输入张量的两部分。
    这是 Gated Linear Unit (GLU) 的一个变体，用于在 Transformer 中
    提升模型表达能力。

    数学公式：
        输入 x 被分成两部分：[x_part, y_part]
        输出 = SiLU(x_part) * y_part
              = x_part * sigmoid(x_part) * y_part

    优势：
    1. 门控机制：y_part 作为门控信号，控制 x_part 的信息流量
    2. 非线性：结合了 SiLU 的平滑非线性和乘法的门控机制
    3. 性能：在 LLaMA、Qwen 等模型中展现出更好的收敛性

    相关概念：
    - SiLU (Sigmoid Linear Unit): f(x) = x * sigmoid(x)
    - GLU (Gated Linear Unit): 使用门控机制的激活函数家族
    - 对比 ReLU：更平滑，梯度流动更好
    """

    def __init__(self):
        """
        初始化 SiluAndMul 激活函数模块

        无需额外参数，该模块是一个纯函数式的激活操作
        """
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：应用 SiLU 和门控乘法

        @torch.compile 装饰器：
            - 使用 TorchCompile 进行图编译优化
            - 将此函数编译为更高效的 CUDA 代码
            - 在推理或微调时显著提升性能（10-30% 加速）

        参数:
            x (torch.Tensor): 输入张量
                            形状: (..., 2*hidden_size)
                            或任何偶数维度的最后一维
                            数据类型: 通常为 float32, float16 或 bfloat16

        返回:
            torch.Tensor: 应用激活后的张量
                         形状: (..., hidden_size)
                         与输入 x 的最后一维相比，缩小为原来的一半

        工作流程:
        1. 将输入张量沿最后一维分成两部分（x_part 和 y_part）
        2. 对 x_part 应用 SiLU 激活函数
        3. 将 SiLU(x_part) 与 y_part 逐元素相乘
        4. 返回结果

        示例:
            >>> activation = SiluAndMul()
            >>> x = torch.randn(batch_size, seq_len, 2*hidden_size)
            >>> output = activation(x)
            >>> output.shape  # (batch_size, seq_len, hidden_size)

        实现细节:
            - chunk(2, -1)：在最后一维上分割成 2 个相等的部分
            - F.silu()：应用 SiLU 激活，公式为 x * sigmoid(x)
            - 逐元素乘法：两个张量对应位置的元素相乘
        """
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
