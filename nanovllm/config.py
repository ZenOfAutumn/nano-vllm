import os
from dataclasses import dataclass

from transformers import AutoConfig


@dataclass
class Config:
    """LLM 推理引擎的核心配置类"""

    model: str
    """
    模型路径：Hugging Face 模型的本地目录路径（必需参数）
    例如：'/path/to/llama-7b' 或 'meta-llama/Llama-2-7b'
    """

    max_num_batched_tokens: int = 16384
    """
    单个批次最大 token 数。
    影响内存使用和吞吐量，值越大吞吐越高但内存占用越大。
    """

    max_num_seqs: int = 512
    """
    单批次最大序列（请求）数。
    限制同时处理的请求数量，影响系统的并发能力。
    """

    max_model_len: int = 4096
    """
    模型的最大上下文长度。
    会被自动约束为不超过 Hugging Face 模型配置中的 max_position_embeddings。
    """

    gpu_memory_utilization: float = 0.9
    """
    GPU 内存使用率（0.0-1.0）。
    设置 GPU 最多可使用的内存比例，0.9 表示使用 90% 的可用 GPU 显存。
    """

    tensor_parallel_size: int = 1
    """
    张量并行大小，必须在 1-8 之间。
    当 > 1 时启用张量并行分布式推理，用于大模型推理加速。
    """

    enforce_eager: bool = False
    """
    强制使用 eager 执行模式。
    True 时禁用图编译优化，False 时允许编译优化以提升性能。
    """

    hf_config: AutoConfig | None = None
    """
    Hugging Face 模型配置对象。
    初始化时自动从 model 参数加载，用于获取模型的元数据（如词表大小、层数等）。
    """

    eos: int = -1
    """
    EOS（End of Sequence）token ID。
    生成时遇到此 token 会停止生成。-1 表示使用模型默认的 EOS token ID。
    """

    kvcache_block_size: int = 256
    """
    KV 缓存块大小，必须是 256 的倍数。
    影响显存中 KV 缓存的粒度管理，值越小碎片化越多但精度越高。
    """

    num_kvcache_blocks: int = -1
    """
    KV 缓存块数量。
    -1 表示自动计算（根据 gpu_memory_utilization 和可用显存），
    正整数表示显式指定块数，影响最大可处理的 token 数。
    """

    def __post_init__(self):
        """dataclass 初始化后自动调用的方法"""
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
