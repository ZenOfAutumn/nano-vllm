"""
旋转位置编码（Rotary Position Embedding, RoPE）实现

RoPE 是一种高效的位置编码方法，通过旋转矩阵将位置信息融入到 Query 和 Key 中。
相比传统的绝对位置编码，RoPE 具有更好的相对位置表示能力。

主要特点：
1. 相对位置偏差性：同一相对距离的任何两个位置对，其旋转矩阵都相同
2. 外推能力强：支持超过训练长度的推理
3. 计算效率高：只需要在 Query 和 Key 上应用旋转变换
"""

from functools import lru_cache

import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    应用旋转位置编码到张量上

    使用旋转矩阵将位置编码应用到输入张量。
    对于 2D 特征分量对 (x1, x2)，通过旋转变换得到 (y1, y2)：
        y1 = x1 * cos(θ) - x2 * sin(θ)
        y2 = x2 * cos(θ) + x1 * sin(θ)

    这相当于对复数进行旋转操作，其中 (x1, x2) 表示复数的实部和虚部。

    参数:
        x (torch.Tensor): 输入张量，最后一维是特征维度，需要被分成两部分
                         形状: (..., head_dim)
        cos (torch.Tensor): 余弦值缓存，包含预计算的 cos(θ) 值
                          形状: (seq_len, 1, head_dim//2)
        sin (torch.Tensor): 正弦值缓存，包含预计算的 sin(θ) 值
                          形状: (seq_len, 1, head_dim//2)

    返回:
        torch.Tensor: 应用旋转后的张量，与输入 x 形状相同

    示例:
        >>> # 假设 head_dim=128
        >>> x = torch.randn(batch_size, seq_len, num_heads, 128)
        >>> cos = torch.randn(seq_len, 1, 64)
        >>> sin = torch.randn(seq_len, 1, 64)
        >>> rotated_x = apply_rotary_emb(x, cos, sin)
        >>> rotated_x.shape == x.shape  # True
    """
    # 将输入张量分成两部分，每部分对应复数的实部和虚部
    # chunk(2, dim=-1) 将最后一维分成两个相等的部分
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)

    # 计算旋转后的第一部分: y1 = x1 * cos(θ) - x2 * sin(θ)
    # 这对应于复数乘法的实部
    y1 = x1 * cos - x2 * sin

    # 计算旋转后的第二部分: y2 = x2 * cos(θ) + x1 * sin(θ)
    # 这对应于复数乘法的虚部
    y2 = x2 * cos + x1 * sin

    # 将两部分拼接回原来的维度
    # 转换回原始数据类型（通常是 float16 或 bfloat16）以节省内存
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码模块

    实现 RoPE (Rotary Position Embedding)，在初始化时预计算所有位置的旋转矩阵。
    通过缓存 cos/sin 值，避免在前向过程中重复计算，大大提高效率。

    数学原理：
    对于位置 m 和特征维度 d，旋转矩阵为：
        R_m = [
            cos(m*θ_0)   -sin(m*θ_0)
            sin(m*θ_0)    cos(m*θ_0)
        ]
    其中 θ_i = base^(-2i/d)，base 通常为 10000

    通过这种旋转变换，相对位置 (i-j) 的信息被编码到旋转角的差异中。
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        """
        初始化旋转位置编码

        参数:
            head_size (int): 注意力头的维度。假设为 128，则每个头有 128 个特征
            rotary_dim (int): 应用旋转编码的维度。必须等于 head_size
            max_position_embeddings (int): 最大位置索引，确定预缓存的序列长度
                                         通常为模型的最大上下文长度（如 4096）
            base (float): 旋转编码的基数。通常为 10000，用于计算 θ_i = base^(-2i/d)

        工作流程:
        1. 计算逆频率 inv_freq：基于基数和维度
        2. 为所有位置 [0, max_position_embeddings) 生成频率矩阵
        3. 计算每个位置-维度对的 cos 和 sin 值
        4. 将它们缓存在 GPU 显存中以供重复使用
        """
        super().__init__()
        self.head_size = head_size

        # 确保旋转维度等于头维度（通常都这样配置）
        assert rotary_dim == head_size

        # 【关键步骤 1】计算逆频率
        # 生成维度索引: [0, 2, 4, ..., rotary_dim-2]（只取偶数索引）
        # 原因：特征分量两两配对进行旋转，所以只需要 rotary_dim/2 个频率值
        # inv_freq = 1.0 / (base^(2i/d)) 其中 i ∈ {0, 1, 2, ..., d/2-1}
        # 这产生了一组指数递增的频率，用于多尺度位置编码
        # 形状: (rotary_dim // 2,)
        # 例如 head_size=128, base=10000:
        #   inv_freq[0] = 1.0 / 10000^(0/128) = 1.0
        #   inv_freq[1] = 1.0 / 10000^(2/128) ≈ 0.9999
        #   inv_freq[63] = 1.0 / 10000^(126/128) ≈ 0.0001
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))

        # 【关键步骤 2】生成位置索引张量
        # 从 0 到 max_position_embeddings-1 的所有位置
        # 这代表序列中每个 token 的位置
        # 形状: (max_position_embeddings,)
        t = torch.arange(max_position_embeddings, dtype=torch.float)

        # 【关键步骤 3】计算频率矩阵
        # freqs[m, i] = t[m] * inv_freq[i] = m * θ_i
        # 这是位置 m 在频率 θ_i 上的旋转角
        # einsum("i,j -> ij") 执行外积：t (i,) × inv_freq (j,) → freqs (i,j)
        # 形状: (max_position_embeddings, rotary_dim // 2)
        # 例如: freqs[5, 10] = 5 * inv_freq[10] = 5 * θ_10
        freqs = torch.einsum("i,j -> ij", t, inv_freq)

        # 【关键步骤 4】计算 cos 和 sin 缓存
        # cos[m, i] = cos(freqs[m, i]) = cos(m * θ_i)
        # sin[m, i] = sin(freqs[m, i]) = sin(m * θ_i)
        # 形状: (max_position_embeddings, rotary_dim // 2)
        cos = freqs.cos()
        sin = freqs.sin()

        # 【关键步骤 5】将 cos 和 sin 拼接并注册为缓冲区
        # 在最后一维上拼接：[cos, sin]
        # 形状变化：(max_pos, rotary_dim//2) + (max_pos, rotary_dim//2)
        #         → (max_pos, rotary_dim)
        # unsqueeze_(1) 在第二维插入大小为 1 的维度，用于广播
        # 最终形状: (max_position_embeddings, 1, rotary_dim)
        # 这个形状便于后续与注意力头进行广播计算
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)

        # 注册为模型缓冲区（不是参数，不参与梯度计算）
        # persistent=False：在保存模型时不保存这个缓冲区（因为可以重新计算）
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        应用旋转位置编码到 Query 和 Key

        @torch.compile 装饰器：使用 TorchCompile 进行图编译优化
        可以将此函数编译为更高效的 CUDA 代码，提升推理速度（10-20% 提升）

        参数:
            positions (torch.Tensor): 每个 token 的位置索引
                                    形状: (batch_size, seq_len) 或 (total_tokens,)
                                    值域: [0, max_position_embeddings)
                                    示例: [[0, 1, 2], [0, 1, 2]] 表示两个序列
            query (torch.Tensor): Query 张量
                                形状: (batch_size, seq_len, num_heads, head_dim)
                                或 (total_tokens, num_heads, head_dim)
            key (torch.Tensor): Key 张量
                              形状同 query

        返回:
            tuple[torch.Tensor, torch.Tensor]: (rotated_query, rotated_key)
                                             都是应用了位置编码后的张量

        工作流程:
        1. 从缓存中查找当前位置的 cos/sin 值
        2. 分离 cos 和 sin
        3. 将旋转变换应用到 Query 和 Key
        4. 返回编码后的张量

        示例:
        >>> positions = torch.tensor([0, 1, 2, 3, 4])  # 5 个位置
        >>> query = torch.randn(5, 8, 128)  # 5 个位置, 8 个头, 128 维
        >>> key = torch.randn(5, 8, 128)
        >>> rope = RotaryEmbedding(head_size=128, rotary_dim=128,
        ...                        max_position_embeddings=2048, base=10000)
        >>> q_rot, k_rot = rope(positions, query, key)
        >>> q_rot.shape == query.shape  # True
        """
        # 【步骤 1】从缓存中查找所需位置的 cos/sin 值
        # cos_sin_cache 形状: (max_position_embeddings, 1, rotary_dim)
        # 通过 positions 索引后，获取这些位置对应的 cos 和 sin 值
        # cos_sin 形状: (batch_size, seq_len, 1, rotary_dim) 或类似形状，取决于 positions
        # 例如: positions=[0, 1, 2] → cos_sin 是第 0, 1, 2 行的缓存数据
        cos_sin = self.cos_sin_cache[positions]

        # 【步骤 2】将 cos 和 sin 从 [cos, sin] 的拼接形式中分离出来
        # chunk(2, dim=-1) 在最后一维上分割成两个相等的部分
        # 原始 cos_sin 形状: (..., rotary_dim) 其中前 rotary_dim//2 是 cos，后面是 sin
        # 分割后:
        #   cos 形状: (..., rotary_dim // 2)
        #   sin 形状: (..., rotary_dim // 2)
        cos, sin = cos_sin.chunk(2, dim=-1)

        # 【步骤 3】应用旋转变换到 Query
        # apply_rotary_emb 函数实现了：
        #   q_rot = (q1*cos - q2*sin, q2*cos + q1*sin)
        # 其中 (q1, q2) 是特征维的两个分量
        query = apply_rotary_emb(query, cos, sin)

        # 【步骤 4】应用相同的旋转变换到 Key
        # Key 使用相同的位置编码确保 Query 和 Key 在相对位置上对齐
        # 这是 RoPE 的关键特性：相对位置信息被编码在旋转角的差异中
        key = apply_rotary_emb(key, cos, sin)

        # 【步骤 5】返回编码后的 Query 和 Key
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    """
    获取或创建 RotaryEmbedding 实例（带缓存）

    @lru_cache(1) 装饰器：
        - 使用最近最少使用 (LRU) 策略缓存函数返回值
        - maxsize=1：最多缓存 1 个结果，因为通常只需要一个 RoPE 配置
        - 这样避免重复创建相同配置的 RotaryEmbedding 对象
        - 内存节省：只保留一个预计算的 cos/sin 缓存

    参数:
        head_size (int): 注意力头的维度（如 128）
        rotary_dim (int): 旋转编码维度（通常等于 head_size）
        max_position (int): 最大位置索引（如 2048 或 4096）
        base (float): 旋转频率的基数（默认 10000）
        rope_scaling (dict | None): 位置外推方法配置
                                   目前不支持（必须为 None）
                                   将来可能支持：
                                   - linear scaling：简单线性扩展
                                   - dynamic ntk：动态 NTK 缩放
                                   - yarn：Yet Another RoPE extensioN

    返回:
        RotaryEmbedding: 旋转位置编码对象，包含预计算的 cos/sin 缓存

    示例:
        >>> # 第一次调用会创建新的 RotaryEmbedding 实例
        >>> rope1 = get_rope(head_size=128, rotary_dim=128,
        ...                  max_position=4096, base=10000)
        >>>
        >>> # 第二次使用相同参数调用会返回缓存的同一实例
        >>> rope2 = get_rope(head_size=128, rotary_dim=128,
        ...                  max_position=4096, base=10000)
        >>> rope1 is rope2  # True，因为使用了 @lru_cache
        >>>
        >>> # 不同参数会创建新实例（但只保存一个在内存中）
        >>> rope3 = get_rope(head_size=256, rotary_dim=256,
        ...                  max_position=8192, base=10000)
        >>> rope1 is rope3  # False，因为参数不同
        >>> rope2 is rope3  # False，rope2 已被淘汰

    工作流程:
    1. 检查是否使用了位置缩放（目前不支持）
    2. 创建 RotaryEmbedding 实例
    3. 返回实例给调用者
    4. 缓存此结果以供后续相同参数的调用
    """
    # 【验证】检查是否传递了不支持的参数
    # 目前 rope_scaling 功能尚未实现
    # 如果需要支持位置外推，需要修改此函数和 RotaryEmbedding 类
    assert rope_scaling is None

    # 【创建】实例化旋转位置编码对象
    # RotaryEmbedding.__init__ 会执行以下操作：
    # 1. 计算所有位置的频率值
    # 2. 预计算 cos 和 sin 值
    # 3. 将它们存储在 GPU 显存中的缓冲区
    # 这是一个一次性的初始化成本，后续调用只需查表
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)

    # 【返回】返回创建好的实例
    # 此实例会被 @lru_cache 缓存
    # 下次使用相同参数调用时，直接返回这个实例，无需重复初始化
    return rotary_emb
