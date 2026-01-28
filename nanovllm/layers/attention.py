"""
Flash Attention 注意力机制实现

实现高效的 Flash Attention 注意力计算，支持 KV 缓存管理。
支持 Prefill 和 Decode 两个推理阶段的不同计算方式。
"""

import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from torch import nn

from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """
    Triton 内核：将 Key 和 Value 存储到 KV 缓存

    这是一个高性能的 CUDA 内核，用于在 Triton 框架下高效地将
    当前生成的 Key 和 Value 存储到预分配的 KV 缓存中。

    工作原理：
    1. 每个线程块处理一个 token 的 Key 和 Value
    2. 根据 slot_mapping 确定缓存中的存储位置
    3. 从输入张量读取 KV，写入缓存张量

    参数:
        key_ptr: Key 张量的指针
        key_stride: Key 张量在第 0 维的步长
        value_ptr: Value 张量的指针
        value_stride: Value 张量在第 0 维的步长
        k_cache_ptr: K 缓存的指针
        v_cache_ptr: V 缓存的指针
        slot_mapping_ptr: 槽位映射的指针
                        slot_mapping[i] 指定第 i 个 token 的 KV 存储位置
                        -1 表示该位置无需存储（e.g., prefill 前的部分）
        D: 特征维度（常量表达式），= num_heads * head_dim

    内核逻辑:
    1. 获取当前线程块的 ID（处理的 token 索引）
    2. 查询槽位映射，获取存储位置
    3. 如果槽位为 -1，跳过处理（可能是前缀缓存的部分）
    4. 计算输入和缓存中的偏移
    5. 加载 Key 和 Value
    6. 存储到缓存中

    性能特性:
    - 内存合并访问：顺序读写提高带宽利用率
    - 无分支发散：所有线程的操作基本相同
    - 最小同步开销：适合批量操作
    """
    # 获取当前线程块处理的 token 索引
    idx = tl.program_id(0)

    # 加载该 token 对应的缓存槽位
    slot = tl.load(slot_mapping_ptr + idx)

    # 如果槽位为 -1，表示无需缓存，直接返回
    if slot == -1:
        return

    # 计算输入张量中的偏移
    # 考虑到每个 token 可能有不同的步长
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    # 从输入张量中加载 Key 和 Value
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # 计算缓存中的偏移
    # 缓存是线性存储的，根据 slot 计算位置
    cache_offsets = slot * D + tl.arange(0, D)

    # 存储到 KV 缓存
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """
    将 Key 和 Value 存储到 KV 缓存（PyTorch 包装函数）

    这是 store_kvcache_kernel 的 PyTorch 包装，负责参数验证和内核调度。

    参数:
        key (torch.Tensor): Key 张量
                          形状: (num_tokens, num_heads, head_dim)
                          内存格式: contiguous，最后一维步长为 1
        value (torch.Tensor): Value 张量
                            形状: (num_tokens, num_heads, head_dim)
                            内存格式: contiguous，最后一维步长为 1
        k_cache (torch.Tensor): K 缓存张量
                              形状: (num_blocks, num_heads, block_size, head_dim)
                              用于存储所有已生成 token 的 Key
        v_cache (torch.Tensor): V 缓存张量
                              形状: (num_blocks, num_heads, block_size, head_dim)
                              用于存储所有已生成 token 的 Value
        slot_mapping (torch.Tensor): 槽位映射
                                   形状: (num_tokens,)
                                   值: 缓存中的绝对位置索引，-1 表示跳过

    工作流程:
    1. 解析张量维度和步长
    2. 验证内存布局的正确性
    3. 调度 Triton 内核执行存储操作

    内存布局要求:
    - Key/Value 的最后一维必须是连续的（步长=1）
    - 第二维（head 维）步长必须是 head_dim
    - 缓存张量的第二维步长必须是总特征维度 D

    示例:
        >>> num_tokens, num_heads, head_dim = 32, 8, 128
        >>> key = torch.randn(num_tokens, num_heads, head_dim).cuda()
        >>> value = torch.randn(num_tokens, num_heads, head_dim).cuda()
        >>> k_cache = torch.zeros(num_blocks, num_heads, block_size, head_dim).cuda()
        >>> v_cache = torch.zeros(num_blocks, num_heads, block_size, head_dim).cuda()
        >>> slot_mapping = torch.arange(num_tokens).cuda()
        >>> store_kvcache(key, value, k_cache, v_cache, slot_mapping)
    """
    # 获取张量维度
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim

    # 验证 Key 和 Value 的内存布局
    # 最后一维必须是连续的（步长为 1）
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    # 第二维（head）步长必须是 head_dim
    assert key.stride(1) == head_dim and value.stride(1) == head_dim

    # 验证缓存的内存布局
    # 第二维步长必须是总特征维度 D
    assert k_cache.stride(1) == D and v_cache.stride(1) == D

    # 验证槽位映射大小与 token 数相同
    assert slot_mapping.numel() == N

    # 调度 Triton 内核
    # [(N,)] 表示启动 N 个线程块，每个处理一个 token
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    """
    Flash Attention 注意力层

    实现高效的多头自注意力机制，使用 Flash Attention 算法。
    支持分组查询注意力（GQA）并集成 KV 缓存管理。

    特点：
    1. Flash Attention：内存高效的注意力计算
    2. GQA 支持：可配置的查询头数和 KV 头数
    3. KV 缓存管理：支持 Prefill 和 Decode 两个阶段
    4. 因果掩码：自注意力只能看到当前和之前的 token
    5. 块级缓存：使用预分配的 KV 缓存块存储历史 token 的 KV

    Prefill vs Decode：
    - Prefill：处理整个输入序列，支持前缀缓存
    - Decode：逐个生成新 token，完全依赖 KV 缓存加速
    """

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        """
        初始化 Flash Attention 层

        参数:
            num_heads (int): 查询头数
                           例如：32（在 TP size=8 的情况下，全局 256）
            head_dim (int): 每个头的维度
                          例如：128
            scale (float): 注意力缩放因子
                         通常为 1 / sqrt(head_dim)
                         例如：0.0883（对于 head_dim=128）
            num_kv_heads (int): 键值头数（支持 GQA）
                              例如：8（对于 GQA 配置）
                              如果 num_heads == num_kv_heads，则为标准 MHA

        初始化内容:
            - 注意力头数和维度参数
            - KV 缓存张量（初始化为空，后续由 BlockManager 填充）

        说明:
            当 num_kv_heads < num_heads 时，实现分组查询注意力（GQA）。
            每个 KV 头对应多个查询头（比例为 num_heads / num_kv_heads）。
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads

        # 初始化 KV 缓存为空张量
        # 后续会由 BlockManager.allocate_kv_cache() 分配真实的缓存张量
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        前向传播：执行多头自注意力计算

        参数:
            q (torch.Tensor): Query 张量
                            形状：(total_tokens, num_heads, head_dim)
                            - total_tokens 是当前批次的总 token 数
                            - 对于 Prefill：是所有序列 token 总数
                            - 对于 Decode：是 batch_size

            k (torch.Tensor): Key 张量
                            形状同 Query
                            在 Prefill 中包含所有当前 token 的 key
                            在 Decode 中只包含新生成的 token 的 key

            v (torch.Tensor): Value 张量
                            形状同 Query
                            在 Prefill 中包含所有当前 token 的 value
                            在 Decode 中只包含新生成的 token 的 value

        返回:
            torch.Tensor: 注意力输出
                         形状：(total_tokens, num_heads, head_dim)
                         对应每个 query token 的注意力加权结果

        工作流程:
        1. 获取执行上下文（包含 KV 缓存槽位、序列长度等）
        2. 如果 KV 缓存已分配，存储新的 Key 和 Value
        3. 根据推理阶段选择不同的计算方式：
           a. Prefill：使用 flash_attn_varlen_func 处理可变长度序列
           b. Decode：使用 flash_attn_with_kvcache 利用缓存加速
        4. 返回注意力输出

        执行阶段详解:

        【Prefill 阶段】:
        - 特征：处理新输入的所有 token
        - 输入：Q, K, V 都是当前输入的 token（通常是新 token）
        - KV 缓存：存储当前生成的 K, V 用于后续 Decode
        - 前缀缓存：如果有前缀（block_tables 不为 None），使用缓存中的历史 KV
        - 计算：对所有 query 计算注意力，支持批处理多个序列

        【Decode 阶段】:
        - 特征：生成新 token，每次生成一个
        - 输入：Q 是当前生成的 token，K, V 从缓存读取
        - KV 缓存：Q 在新 token，注意力使用所有历史 KV（包括新 token）
        - 优化：新 token 只需与所有历史 token 计算注意力，不需重新计算
        - 计算：高度并行化，单个 token 但非常高效

        性能特性:
        - Prefill：IO 受限，适合批处理
        - Decode：计算受限，受益于 KV 缓存
        - KV 缓存：块级存储，支持内存高效的序列管理

        示例：
            >>> attn = Attention(num_heads=8, head_dim=128, scale=0.0883, num_kv_heads=8)
            >>> # Decode 阶段
            >>> q = torch.randn(32, 8, 128).cuda()  # 32 个 token，8 个头
            >>> k = torch.randn(32, 8, 128).cuda()
            >>> v = torch.randn(32, 8, 128).cuda()
            >>> output = attn(q, k, v)
            >>> output.shape  # torch.Size([32, 8, 128])
        """
        # 获取执行上下文（包含 KV 缓存信息、序列长度等）
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # 【步骤 1】如果 KV 缓存已分配，存储新的 Key 和 Value
        # 这在每个推理步骤中执行（包括 Prefill 和 Decode）
        # 存储当前 token 的 KV，以便后续 Decode 阶段使用
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # 【步骤 2】根据推理阶段选择不同的计算方式
        if context.is_prefill:
            # ========== Prefill 阶段 ==========
            # 处理新输入序列的所有 token，计算完整的自注意力

            # 如果有前缀缓存（block_tables 不为 None），使用缓存中的历史 KV
            # 这支持处理模型中已生成的前缀部分
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache

            # 使用 Flash Attention 处理可变长度序列
            # flash_attn_varlen_func 支持批量处理多个不同长度的序列
            # 通过 cu_seqlens_q/k 指定每个序列的边界
            # causal=True 启用因果掩码（只能看到当前和之前的 token）
            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q,     # 查询最大长度
                cu_seqlens_q=context.cu_seqlens_q,     # 查询累积长度
                max_seqlen_k=context.max_seqlen_k,     # 键最大长度
                cu_seqlens_k=context.cu_seqlens_k,     # 键累积长度
                softmax_scale=self.scale,               # 注意力缩放
                causal=True,                            # 因果掩码
                block_table=context.block_tables        # 块表（用于前缀缓存）
            )
        else:
            # ========== Decode 阶段 ==========
            # 生成单个新 token，利用 KV 缓存快速计算

            # flash_attn_with_kvcache 期望 query 的形状为 (batch_size, seq_len=1, ...)
            # unsqueeze(1) 将 (batch_size, num_heads, head_dim) 变为 (batch_size, 1, num_heads, head_dim)
            # cache_seqlens 指定每个序列的当前长度（用于计算相对位置）
            # block_table 指定每个序列使用的 KV 缓存块
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),                         # 添加 seq_len 维度
                k_cache,                                 # 历史 key 缓存
                v_cache,                                 # 历史 value 缓存
                cache_seqlens=context.context_lens,     # 每个序列的当前长度
                block_table=context.block_tables,        # KV 缓存块表
                softmax_scale=self.scale,               # 注意力缩放
                causal=True                             # 因果掩码
            )

        # 返回注意力输出
        return o
