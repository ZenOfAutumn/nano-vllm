"""
Qwen3 语言模型实现

该模块包含完整的 Qwen3 模型架构，包括：
1. Qwen3Attention - 多头注意力层（支持 GQA、张量并行、KV 缓存）
2. Qwen3MLP - 前馈网络层（MLP）
3. Qwen3DecoderLayer - Transformer 解码器层
4. Qwen3Model - 完整的 Qwen3 模型主体
5. Qwen3ForCausalLM - 用于因果语言建模的完整模型（带权重映射）

核心特性：
- 支持分组查询注意力（GQA）：多个 Query 头共享一个 Key/Value 头
- 张量并行：通过 dist 将模型分片到多个 GPU
- 旋转位置编码（RoPE）：相对位置编码
- RMSNorm：Layer Normalization 替代
- 闪电注意力（Flash Attention）优化
- KV 缓存支持：高效存储键值对
"""

import torch
import torch.distributed as dist
from torch import nn
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import , MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope


class Qwen3Attention(nn.Module):
    """
    Qwen3 多头自注意力层（Self-Attention）。

    支持分组查询注意力（GQA）和张量并行分布式推理。
    在 Transformer 中，注意力机制允许模型关注输入序列中不同位置的信息。

    GQA 是对多头注意力的优化：
    - 传统 MHA：所有 Query 头都有独立的 Key/Value 头（参数多，显存占用大）
    - GQA：多个 Query 头共享一个 Key/Value 头（参数少，显存占用小）

    公式：
        Attention(Q, K, V) = softmax(Q·K^T / √d_k)·V
    其中：
        Q: Query 矩阵 (batch, seq_len, d_k)
        K: Key 矩阵 (batch, seq_len, d_k)
        V: Value 矩阵 (batch, seq_len, d_v)
        d_k: Key 的维度（head_dim）
        √d_k: 缩放因子，保持 softmax 分布稳定
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        """
        初始化注意力层。

        Args:
            hidden_size: 隐藏状态维度（如 4096）
            num_heads: 总 Query 头数（如 64）
            num_kv_heads: 总 Key/Value 头数（如 8，用于 GQA）
            max_position: 最大序列长度（如 131,072）
            head_dim: 每个头的维度（如 64），默认为 hidden_size // num_heads
            rms_norm_eps: RMSNorm 的 epsilon 值（数值稳定性）
            qkv_bias: 是否在 Q/K/V 投影中使用偏置
            rope_theta: RoPE 位置编码的基数（频率参数）
            rope_scaling: RoPE 的缩放参数（用于扩展上下文长度）

        流程：
            1. 初始化张量并行大小
            2. 根据张量并行分割 Q 头和 KV 头
            3. 创建 Q/K/V 投影层、输出投影层
            4. 初始化旋转位置编码（RoPE）
            5. 如果不使用 bias，创建 Q/K 规范化层
        """
        super().__init__()

        # ===== 张量并行相关 =====
        # 获取分布式进程组的世界大小（GPU 数量）
        tp_size = dist.get_world_size()

        # ===== Query 头相关 =====
        # 总 Query 头数（未分割）
        self.total_num_heads = num_heads
        # 断言：Query 头数必须能被张量并行大小整除
        assert self.total_num_heads % tp_size == 0
        # 当前 GPU 上的 Query 头数
        self.num_heads = self.total_num_heads // tp_size

        # ===== Key/Value 头相关 =====
        # 总 Key/Value 头数（GQA 中通常远小于 Query 头数）
        self.total_num_kv_heads = num_kv_heads
        # 断言：KV 头数必须能被张量并行大小整除
        assert self.total_num_kv_heads % tp_size == 0
        # 当前 GPU 上的 Key/Value 头数
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        # ===== 头维度相关 =====
        # 每个头的维度（通常 hidden_size / num_heads，如 4096 / 64 = 64）
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        # Query 的总维度（所有 Query 头的维度之和）
        self.q_size = self.num_heads * self.head_dim
        # Key/Value 的总维度（所有 KV 头的维度之和）
        self.kv_size = self.num_kv_heads * self.head_dim

        # ===== 注意力缩放因子 =====
        # 缩放因子 = 1/√(head_dim)，用于防止 softmax 梯度消失
        # 数学推导：当 head_dim 增大时，Q·K^T 的方差增大，导致 softmax 饱和
        # 通过除以 √d_k 来保持方差为 1
        self.scaling = self.head_dim ** -0.5

        # ===== Bias 配置 =====
        # 是否在 QKV 投影中使用偏置
        self.qkv_bias = qkv_bias

        # ===== Q/K/V 投影层 =====
        # QKVParallelLinear：同时计算 Q、K、V 的投影
        # 支持列并行：输出被分割到不同 GPU
        # 输入：hidden_size，输出：q_size + kv_size + kv_size
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )

        # ===== 输出投影层 =====
        # RowParallelLinear：支持行并行
        # 将注意力输出从多头形式投影回 hidden_size
        # 输入：num_heads * head_dim，输出：hidden_size
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )

        # ===== 旋转位置编码（RoPE）=====
        # RoPE 通过旋转 Q 和 K 来编码相对位置信息
        # 比绝对位置编码更灵活，支持上下文长度外推
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,  # 频率基数
            rope_scaling=rope_scaling,  # 上下文长度扩展
        )

        # ===== 注意力计算层 =====
        # 封装的注意力计算（支持 Flash Attention、KV 缓存等）
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

        # ===== Query/Key 规范化层 =====
        # 如果不使用 bias，则对 Q 和 K 进行 RMSNorm
        # 这是 Qwen 的创新：对原始 Q/K 进行规范化，而不是对 QK^T 的结果规范化
        if not self.qkv_bias:
            # Q 的规范化层
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            # K 的规范化层
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播：执行自注意力计算。

        Args:
            positions: 位置索引张量，形状 (seq_len,)，用于 RoPE
            hidden_states: 输入隐藏状态，形状 (seq_len, hidden_size)

        Returns:
            注意力输出，形状 (seq_len, hidden_size)

        流程：
            1. 投影：hidden_states → Q, K, V
            2. Reshape：将线性投影结果分解为多头形式
            3. 规范化：对 Q、K 进行 RMSNorm（如果启用）
            4. 位置编码：应用 RoPE 到 Q、K
            5. 注意力：计算 Attention(Q, K, V)
            6. 输出投影：多头输出投影回 hidden_size
        """
        # ===== 步骤 1：Q/K/V 投影 =====
        # 一次性计算 Q, K, V，提高效率
        # 输出形状：(seq_len, q_size + kv_size + kv_size)
        qkv = self.qkv_proj(hidden_states)

        # ===== 步骤 2：分离和 Reshape Q/K/V =====
        # 按维度分割 qkv 张量
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Reshape Q：(seq_len, q_size) → (seq_len, num_heads, head_dim)
        q = q.view(-1, self.num_heads, self.head_dim)

        # Reshape K：(seq_len, kv_size) → (seq_len, num_kv_heads, head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)

        # Reshape V：(seq_len, kv_size) → (seq_len, num_kv_heads, head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # ===== 步骤 3：Query/Key 规范化 =====
        # 如果没有使用 bias，则对 Q、K 应用 RMSNorm
        # 这个设计是 Qwen3 的特性
        if not self.qkv_bias:
            # 对每个头的 Query 进行规范化
            q = self.q_norm(q)
            # 对每个头的 Key 进行规范化
            k = self.k_norm(k)

        # ===== 步骤 4：旋转位置编码（RoPE）=====
        # 应用 RoPE 到 Q 和 K，将相对位置信息融入向量中
        # 输出：rotated Q 和 K，形状不变
        q, k = self.rotary_emb(positions, q, k)

        # ===== 步骤 5：注意力计算 =====
        # 计算 softmax(Q·K^T / √d_k)·V
        # 支持 Flash Attention、KV 缓存等优化
        # 输出形状：(seq_len, num_heads, head_dim)
        o = self.attn(q, k, v)

        # ===== 步骤 6：展平和输出投影 =====
        # 将多头输出展平：(seq_len, num_heads, head_dim) → (seq_len, q_size)
        output = self.o_proj(o.flatten(1, -1))

        # 最终输出形状：(seq_len, hidden_size)
        return output


class Qwen3MLP(nn.Module):
    """
    Qwen3 前馈网络层（MLP - Multilayer Perceptron）。

    Transformer 中的 MLP 通常由两个线性层组成，中间使用非线性激活函数。
    Qwen3 使用了 "gated" MLP 结构，其中通过门控机制控制信息流。

    结构：
        Input → Gate-Up 投影 → SiLU & Mul → Down 投影 → Output

    其中：
        - Gate-Up 投影：同时计算两个路径的线性变换
          输入维度：hidden_size
          输出维度：2 * intermediate_size（一半用作 gate，一半用作 input）
        - SiLU & Mul：选择性线性单元 = SiLU(x) * y
          其中 x 是 gate，y 是输入
        - Down 投影：将 intermediate_size 投影回 hidden_size

    示例（hidden_size=4096, intermediate_size=11008）：
        Input: (seq_len, 4096)
        → Gate-Up: (seq_len, 22016)
        → Split: gate=(seq_len, 11008), value=(seq_len, 11008)
        → SiLU & Mul: (seq_len, 11008)
        → Down: (seq_len, 4096)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        """
        初始化 MLP 层。

        Args:
            hidden_size: 隐藏状态维度（如 4096）
            intermediate_size: 中间层维度（通常 hidden_size * 4/3，如 11008）
                               这是 MLP 的"宽度"，决定参数量和计算量
            hidden_act: 激活函数类型，当前仅支持 "silu"（Sigmoid Linear Unit）

        流程：
            1. 创建 Gate-Up 投影（列并行，同时计算 gate 和 value）
            2. 创建 Down 投影（行并行）
            3. 验证激活函数为 SiLU
            4. 创建激活函数实例
        """
        super().__init__()

        # ===== Gate-Up 投影层 =====
        # MergedColumnParallelLinear：两个投影合并在一起
        # 这样可以：
        # 1. 减少通信次数（一次发送多个投影的结果）
        # 2. 提高内存访问效率（共享缓冲区）
        # 输入：(seq_len, hidden_size)
        # 输出：(seq_len, intermediate_size * 2)
        # 其中前 intermediate_size 用作 gate，后 intermediate_size 用作 value
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,  # 两个输出都是 intermediate_size
            bias=False,  # 通常 Transformer 中不使用 bias
        )

        # ===== Down 投影层 =====
        # RowParallelLinear：行并行
        # 将 MLP 的输出投影回原始维度
        # 输入：(seq_len, intermediate_size)
        # 输出：(seq_len, hidden_size)
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

        # ===== 激活函数验证 =====
        # 确保激活函数是 SiLU（Sigmoid Linear Unit）
        # SiLU(x) = x * sigmoid(x)，性能优于 ReLU 和 GELU
        assert hidden_act == "silu"

        # ===== 激活函数实例 =====
        # SiluAndMul：结合 SiLU 激活和门控乘法
        # 计算：gate_output * value_output，其中 gate_output 先过 SiLU
        self.act_fn = SiluAndMul()

    def forward(self, x):
        """
        前向传播：执行 MLP 计算。

        Args:
            x: 输入张量，形状 (seq_len, hidden_size)

        Returns:
            MLP 输出，形状 (seq_len, hidden_size)

        流程：
            1. Gate-Up 投影：hidden_size → intermediate_size * 2
            2. SiLU & Mul：gate * value，得到 intermediate_size
            3. Down 投影：intermediate_size → hidden_size
        """
        # ===== 步骤 1：Gate-Up 投影 =====
        # 同时计算两个路径
        # 输出：(seq_len, intermediate_size * 2)
        gate_up = self.gate_up_proj(x)

        # ===== 步骤 2：SiLU 激活 + 门控乘法 =====
        # SiluAndMul 内部将 gate_up 分为两部分：
        # 前一半作为 gate，后一半作为 value
        # 计算：SiLU(gate) * value
        # 输出：(seq_len, intermediate_size)
        x = self.act_fn(gate_up)

        # ===== 步骤 3：Down 投影 =====
        # 投影回原始维度
        # 输出：(seq_len, hidden_size)
        x = self.down_proj(x)

        return x


class Qwen3DecoderLayer(nn.Module):
    """
    Qwen3 Transformer 解码器层。

    标准 Transformer 解码器层包含：
    1. 自注意力（Self-Attention）
    2. 前馈网络（Feed-Forward Network）
    每个子层都有残差连接（Residual Connection）和层归一化（LayerNorm）。

    Qwen3 采用的架构：前置规范化（Pre-Norm）
        Input → LayerNorm → Self-Attention → Residual Add
               → LayerNorm → MLP → Residual Add
               → Output

    相比后置规范化（Post-Norm）的优点：
    - 更稳定的梯度流动
    - 更快的收敛速度
    - 支持更深的网络
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        """
        初始化解码器层。

        Args:
            config: Qwen3 模型配置对象，包含：
                - hidden_size: 隐藏维度
                - num_attention_heads: 总 Query 头数
                - num_key_value_heads: 总 KV 头数
                - max_position_embeddings: 最大序列长度
                - rms_norm_eps: RMSNorm epsilon
                - intermediate_size: MLP 中间层维度
                - hidden_act: 激活函数类型

        流程：
            1. 创建自注意力层
            2. 创建 MLP 层
            3. 创建两个 RMSNorm 层（Pre-Norm）
        """
        super().__init__()

        # ===== 自注意力层 =====
        # 多头自注意力，支持 GQA 和张量并行
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        # ===== MLP 层 =====
        # 前馈网络
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

        # ===== 输入层归一化 =====
        # 应用于注意力之前
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # ===== 注意力后层归一化 =====
        # 应用于 MLP 之前
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：执行解码器层的计算。

        Args:
            positions: 位置索引，形状 (seq_len,)，用于 RoPE
            hidden_states: 隐藏状态，形状 (seq_len, hidden_size)
            residual: 残差张量（来自前一层），形状 (seq_len, hidden_size)
                      第一层为 None

        Returns:
            (hidden_states, residual) 元组
            - hidden_states: 当前层的输出，形状 (seq_len, hidden_size)
            - residual: 用于下一层的残差，形状 (seq_len, hidden_size)

        流程（Pre-Norm 架构）：
            1. 如果 residual 为 None，创建初始残差（等于 hidden_states）
            2. 规范化 hidden_states
            3. 自注意力
            4. 残差连接
            5. 规范化
            6. MLP
            7. 残差连接
            8. 返回输出和新的残差

        残差连接的优势：
        - 梯度可以直接流向前面的层（缓解梯度消失）
        - 参数更新更稳定
        - 支持更深的网络
        """
        # ===== 步骤 1-2：初始化残差，规范化输入 =====
        if residual is None:
            # 第一层：residual 初始化为原始 hidden_states
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 后续层：使用 RMSNorm 的高效残差连接
            # RMSNorm 支持直接处理残差，避免额外的加法操作
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # ===== 步骤 3：自注意力 =====
        # 计算：Attention(Q, K, V)
        # 输出形状：(seq_len, hidden_size)
        hidden_states = self.self_attn(positions, hidden_states)

        # ===== 步骤 4-5：注意力后规范化和残差 =====
        # 在应用 MLP 之前再次规范化
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # ===== 步骤 6：MLP =====
        # 前馈网络
        # 输出形状：(seq_len, hidden_size)
        hidden_states = self.mlp(hidden_states)

        # ===== 步骤 7：返回结果 =====
        # 返回当前层输出和残差（用于下一层）
        # 残差在返回前会在下一层通过 RMSNorm 与 hidden_states 相加
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """
    Qwen3 完整的主干模型。

    由以下部分组成：
    1. 词嵌入层（Token Embedding）- 将 token ID 转换为向量
    2. 多个解码器层（Decoder Layers）- 堆叠的 Transformer 层
    3. 最后的层归一化（Final LayerNorm）- 最后一个规范化层

    推理流程：
        token_id → embedding → [decoder_layer_0 → ... → decoder_layer_79] → final_norm → logits

    相比推理优化的轻量级版本（如 Llama.cpp），Qwen3Model 的优点：
    - 完整的 Transformer 实现，易于理解
    - 支持分布式推理（张量并行）
    - 支持 KV 缓存
    - 支持 Flash Attention
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        """
        初始化 Qwen3 模型。

        Args:
            config: Qwen3 配置对象，包含：
                - vocab_size: 词表大小
                - hidden_size: 隐藏维度
                - num_hidden_layers: 层数（如 80）
                - rms_norm_eps: RMSNorm epsilon

        流程：
            1. 创建词嵌入层
            2. 创建 num_hidden_layers 个解码器层
            3. 创建最后的层归一化
        """
        super().__init__()

        # ===== 词嵌入层 =====
        # 将 token ID（0 到 vocab_size-1）映射到 hidden_size 维向量
        # VocabParallelEmbedding 支持张量并行
        # 输入：(seq_len,) token IDs
        # 输出：(seq_len, hidden_size)
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

        # ===== 解码器层堆栈 =====
        # 创建 config.num_hidden_layers 个相同结构的解码器层
        # Qwen3 通常有 80 层
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])

        # ===== 最后的层归一化 =====
        # 在所有解码器层之后的规范化，稳定输出
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播：执行完整的模型推理。

        Args:
            input_ids: 输入 token ID，形状 (seq_len,)，值在 [0, vocab_size)
            positions: 位置索引，形状 (seq_len,)，用于 RoPE 位置编码

        Returns:
            最终隐藏状态，形状 (seq_len, hidden_size)

        流程：
            1. 嵌入：token ID → 向量
            2. 逐层传播：依次通过 80 个解码器层
            3. 最后规范化：稳定输出
            4. 返回隐藏状态（之后会被投影到词表大小以获得 logits）

        优化：
        - 支持 KV 缓存（在解码器层中实现）
        - 支持 Flash Attention（在注意力层中实现）
        - 支持张量并行（在各层中实现）
        """
        # ===== 步骤 1：嵌入 =====
        # token ID → 隐藏向量
        # 输出形状：(seq_len, hidden_size)
        hidden_states = self.embed_tokens(input_ids)

        # ===== 步骤 2：初始化残差 =====
        # 第一层使用 None，后续层使用前一层的输出
        residual = None

        # ===== 步骤 3：通过所有解码器层 =====
        # 依次通过 80 层，每层进行自注意力 + MLP
        for layer in self.layers:
            # 每层返回 (hidden_states, residual)
            # residual 在下一层的规范化中被使用
            hidden_states, residual = layer(positions, hidden_states, residual)

        # ===== 步骤 4：最后的层归一化 =====
        # 在所有解码器层之后进行最后的规范化
        # 输入：hidden_states 和 residual（最后一层的残差）
        # 输出：(seq_len, hidden_size)
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """
    Qwen3 因果语言模型（Causal Language Model）。

    完整的生成模型，包含：
    1. Qwen3Model - 主干模型（Backbone）
    2. LM Head - 语言模型头，将隐藏状态投影到词表大小

    "因果" 的含义：
    - 模型只能关注当前位置及之前的位置，不能看到未来
    - 这样可以进行自回归生成：逐 token 生成

    "LM Head" 是什么：
    - 最后一个线性层：(seq_len, hidden_size) → (seq_len, vocab_size)
    - 输出 logits，代表每个 token 在词表中的得分
    - 通过 softmax + sampling 生成下一个 token

    权重映射（packed_modules_mapping）的用途：
    - 在加载预训练权重时，某些权重被"打包"以提高效率
    - 需要将打包的权重解映射回模型的对应位置
    - 例：HuggingFace 模型可能有 q_proj、k_proj、v_proj，
      但 Qwen3 将它们合并为 qkv_proj
    """

    # ===== 权重打包映射 =====
    # 在从 safetensors 加载权重时使用
    # 格式：{原始权重名: (目标权重名, 分片索引)}
    # 这允许从不同格式的预训练权重加载到 Qwen3 模型
    packed_modules_mapping = {
        # Query/Key/Value 的单独投影被合并为 qkv_proj
        "q_proj": ("qkv_proj", "q"),       # Query 投影映射到 qkv_proj 的 "q" 部分
        "k_proj": ("qkv_proj", "k"),       # Key 投影映射到 qkv_proj 的 "k" 部分
        "v_proj": ("qkv_proj", "v"),       # Value 投影映射到 qkv_proj 的 "v" 部分

        # Gate 和 Up 投影被合并为 gate_up_proj
        "gate_proj": ("gate_up_proj", 0),  # Gate 投影映射到 gate_up_proj 的前半部分
        "up_proj": ("gate_up_proj", 1),    # Up 投影映射到 gate_up_proj 的后半部分
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        """
        初始化因果语言模型。

        Args:
            config: Qwen3 配置对象

        流程：
            1. 创建主干模型（Qwen3Model）
            2. 创建 LM Head（输出投影层）
            3. 如果启用权重绑定，共享嵌入和输出权重
        """
        super().__init__()

        # ===== 主干模型 =====
        # 完整的 Qwen3 Transformer 模型
        self.model = Qwen3Model(config)

        # ===== LM Head（语言模型头）=====
        # 将隐藏状态投影到词表大小
        # ParallelLMHead 支持张量并行
        # 输入：(seq_len, hidden_size)
        # 输出：(seq_len, vocab_size)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

        # ===== 权重绑定（Weight Tying）=====
        # 将嵌入层和输出层的权重共享
        # 优点：
        # 1. 减少参数量
        # 2. 嵌入和输出层学习相同的表示
        # 3. 提高模型效率
        if config.tie_word_embeddings:
            # 让 LM Head 的权重等于嵌入层的权重
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播：从 token ID 到隐藏状态。

        Args:
            input_ids: 输入 token ID，形状 (seq_len,)
            positions: 位置索引，形状 (seq_len,)

        Returns:
            隐藏状态，形状 (seq_len, hidden_size)

        说明：
        - 这个方法返回的是隐藏状态，不是 logits
        - 要获得 logits，需要调用 compute_logits 方法
        - 这种分离设计是为了支持 KV 缓存等优化

        推理流程（完整）：
            1. forward() 得到隐藏状态
            2. compute_logits() 得到 logits
            3. 采样得到下一个 token
        """
        # 委托给主干模型
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        从隐藏状态计算 logits。

        Args:
            hidden_states: 隐藏状态，形状 (seq_len, hidden_size)

        Returns:
            logits，形状 (seq_len, vocab_size)

        用途：
        - 在推理过程中单独调用，避免重复计算隐藏状态
        - 支持 KV 缓存优化：只需要重新计算最后一层

        示例：
            # Prefill 阶段：处理整个 prompt
            hidden_states = model(input_ids, positions)
            logits = model.compute_logits(hidden_states)

            # Decode 阶段：只处理最后一个 token
            hidden_states = model(last_token_id, last_position)
            logits = model.compute_logits(hidden_states)
        """
        # 调用 LM Head 得到词表大小的输出
        return self.lm_head(hidden_states)
