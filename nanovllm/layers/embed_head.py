"""
词表并行嵌入层（Vocabulary Parallel Embedding）

实现分布式环境中的词表分片嵌入层和语言模型头。
支持张量并行(TP)，将词表分散到多个 GPU 上存储和计算。
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """
    词表并行嵌入层

    在多 GPU 张量并行(TP)中，将词表分片到不同 GPU 上。
    每个 GPU 只存储词表的一部分权重，前向传播时通过掩码和全局约化获得完整结果。

    工作原理：
    1. 词表分片：将总词表大小分成 tp_size 个分片，每个 GPU 维护一个分片
    2. 前向传播：根据输入 token ID 的范围判断是否属于本 GPU 的分片
    3. 全局约化：将各 GPU 的嵌入结果相加，得到完整的嵌入表示

    优势：
    - 内存节省：每个 GPU 只需存储 1/tp_size 的词表
    - 计算分布：嵌入查询在多个 GPU 上并行执行
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        """
        初始化词表并行嵌入层

        参数:
            num_embeddings (int): 词表总大小（所有 token 的种类数）
                                例如：Qwen3 通常为 151936
            embedding_dim (int): 嵌入维度（隐藏层维度）
                                例如：Qwen3 通常为 4096 或 5120

        初始化过程:
        1. 获取当前进程的 TP rank 和总 TP size
        2. 计算当前 GPU 需要维护的词表范围
        3. 初始化权重参数（只存储分片部分）
        4. 绑定权重加载函数用于初始化

        示例:
            >>> # 在 4 GPU TP 模式下
            >>> embed = VocabParallelEmbedding(num_embeddings=160000, embedding_dim=4096)
            >>> # GPU 0 维护 0-40000 的词表
            >>> # GPU 1 维护 40000-80000 的词表
            >>> # ...
        """
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings

        # 计算当前 GPU 需要维护的词表大小
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size

        # 计算当前 GPU 维护的词表范围 [vocab_start_idx, vocab_end_idx)
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition

        # 初始化权重：只存储分片部分
        # 形状: (num_embeddings_per_partition, embedding_dim)
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))

        # 绑定权重加载函数，用于在模型初始化时从检查点加载权重
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        权重加载函数（用于从检查点恢复权重）

        从完整的词表权重中提取当前 GPU 需要的分片部分。

        参数:
            param (nn.Parameter): 当前 GPU 的权重参数
                                形状: (num_embeddings_per_partition, embedding_dim)
            loaded_weight (torch.Tensor): 从检查点加载的完整词表权重
                                         形状: (num_embeddings, embedding_dim)

        工作流程:
        1. 计算当前 GPU 需要提取的分片大小
        2. 计算在完整权重中的起始位置
        3. 从完整权重中提取当前 GPU 的分片
        4. 复制到参数中

        示例:
            >>> # 在 4 GPU TP 中，GPU 2 的权重加载
            >>> # loaded_weight 形状: (160000, 4096)
            >>> # 提取第 2 个分片: loaded_weight[80000:120000]
        """
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """
        前向传播：词表并行嵌入查询

        参数:
            x (torch.Tensor): 输入 token ID 张量
                            形状: (...,) 任意维度的张量
                            值域: [0, num_embeddings)
                            例如: (batch_size, seq_len)

        返回:
            torch.Tensor: 嵌入表示张量
                         形状: (..., embedding_dim)
                         例如: (batch_size, seq_len, embedding_dim)

        单 GPU 流程（tp_size=1）:
        1. 直接使用 F.embedding 查询权重

        多 GPU 流程（tp_size>1）:
        1. 根据 token ID 范围创建掩码，标记哪些 token 属于本 GPU
        2. 将本 GPU 外的 token ID 置为 0（避免超出范围）
        3. 在本 GPU 的权重中查询嵌入
        4. 使用掩码将不属于本 GPU 的位置置为 0
        5. 使用 dist.all_reduce 在所有 GPU 间相加，得到完整结果

        示例:
            >>> embed = VocabParallelEmbedding(num_embeddings=160000, embedding_dim=4096)
            >>> token_ids = torch.tensor([[100, 40050, 80000]])  # 2 个 GPU 的情况
            >>> embeddings = embed(token_ids)
            >>> embeddings.shape  # (1, 3, 4096)

        性能特性：
        - 单 GPU：完全等效于标准 nn.Embedding
        - 多 GPU：通信成本为一次 all_reduce 操作
        """
        if self.tp_size > 1:
            # 创建掩码：标记哪些 token 属于当前 GPU 的词表范围
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 将 token ID 转换为本地索引 [0, num_embeddings_per_partition)
            x = mask * (x - self.vocab_start_idx)

        # 查询嵌入权重
        y = F.embedding(x, self.weight)

        if self.tp_size > 1:
            # 使用掩码将不属于本 GPU 的结果置为 0
            y = mask.unsqueeze(1) * y
            # 在所有 GPU 间相加，得到完整的嵌入结果
            # 由于只有属于该 GPU 的位置非零，相加得到完整结果
            dist.all_reduce(y)

        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    并行语言模型头（Parallel Language Model Head）

    继承自 VocabParallelEmbedding，用于生成 token 预测的 logits。
    相比嵌入层的全局约化，LM Head 使用 gather 操作在主进程(rank=0)
    合并各 GPU 的 logits，然后返回完整的词表 logits。

    特点：
    1. Prefill 优化：在 Prefill 阶段只计算序列最后一个 token 的 logits
    2. Decode 优化：Decode 阶段自然只有一个 token，已最优
    3. 张量并行：使用 gather 将各 GPU 的 logits 收集到 rank=0
    4. 无偏置：LM Head 通常不使用偏置项以减少内存占用
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        """
        初始化并行语言模型头

        参数:
            num_embeddings (int): 词表总大小
            embedding_dim (int): 嵌入/隐藏层维度
            bias (bool): 是否使用偏置项，默认 False
                        LM Head 通常不用偏置以节省内存

        注意:
            - 不支持偏置项（bias=False）
            - 权重矩阵会被用作线性变换的权重
        """
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        """
        前向传播：生成 token 预测 logits

        参数:
            x (torch.Tensor): 隐藏状态张量
                            形状（Prefill）: (total_tokens, embedding_dim)
                                          其中 total_tokens 是所有序列 token 总数
                            形状（Decode）: (batch_size, embedding_dim)
                                          因为每个序列只产生一个新 token

        返回:
            torch.Tensor: Logits 张量
                         形状（单 GPU）: (..., num_embeddings)
                         形状（多 GPU）: (batch_size, num_embeddings) 仅在 rank=0
                                        其他 rank 返回 None
                         值：未经 softmax 的原始预测分数

        工作流程（Prefill 阶段）:
        1. 获取执行上下文
        2. 检查是否处于 Prefill 阶段
        3. 提取每个序列的最后一个 token 的隐藏状态
           - cu_seqlens_q[1:] - 1 获取每个序列最后一个 token 的索引
           - 例如：cu_seqlens_q=[0,5,8,12] → [4,7,11] 对应三个序列的最后 token
        4. 计算这些 token 的 logits

        工作流程（Decode 阶段）:
        1. 执行 Prefill 的所有步骤，但 x 已是 (batch_size, embedding_dim)
        2. 无需额外选择，直接计算 logits

        多 GPU 逻辑:
        1. 每个 GPU 只计算其负责词表部分的 logits
        2. rank=0 创建接收缓冲区数组
        3. 使用 dist.gather 收集所有 GPU 的 logits 到 rank=0
        4. rank=0 在最后一维拼接所有 logits，得到完整词表
        5. 只有 rank=0 返回有效 logits，其他 GPU 返回 None

        性能特性：
        - Prefill：只计算序列末尾 token 的 logits，节省计算
        - Decode：自然只有 batch_size 个 logits（一个 token 一个）
        - 多 GPU：gather 操作将多个小 logits 合并为完整词表结果

        示例:
            >>> lm_head = ParallelLMHead(num_embeddings=151936, embedding_dim=4096)
            >>> hidden = torch.randn(32, 4096)  # 32 个 token 的隐藏状态
            >>> logits = lm_head(hidden)
            >>> logits.shape  # (32, 151936) 在 rank=0；(32, 37984) 在其他 rank
        """
        context = get_context()

        if context.is_prefill:
            # 在 Prefill 阶段，只需计算每个序列最后一个 token 的 logits
            # cu_seqlens_q 记录每个序列在 token 序列中的位置
            # 例如：3 个序列长度 [5, 3, 4] → cu_seqlens_q = [0, 5, 8, 12]
            # 最后 token 索引 = [5-1, 8-1, 12-1] = [4, 7, 11]
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()

        # 计算 logits：线性变换
        # 使用权重矩阵（词表分片或完整词表）进行线性变换
        logits = F.linear(x, self.weight)

        if self.tp_size > 1:
            # 多 GPU 情况：需要聚合所有 GPU 的 logits
            # rank=0 创建接收缓冲区，其他 GPU 为 None
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None

            # 将所有 GPU 的 logits gather 到 rank=0
            # 每个 GPU 贡献其负责词表部分的 logits
            dist.gather(logits, all_logits, 0)

            # rank=0 拼接所有 logits，得到完整词表的预测
            # 在最后一维拼接：(B, vocab_part) * tp_size → (B, vocab)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None

        return logits
