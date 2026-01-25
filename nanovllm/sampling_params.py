from dataclasses import dataclass


@dataclass
class SamplingParams:
    """
    LLM 文本生成的采样参数配置类。

    控制模型生成文本时的行为，影响生成质量、多样性和长度。
    这些参数在推理过程中被传递给采样器来控制 token 选择策略。
    """

    temperature: float = 1.0
    """
    温度参数，控制生成的随机性。范围：(0, +∞)，通常在 (0, 2] 之间。

    - temperature < 1.0：降低随机性，模型更倾向选择概率高的 token，生成更确定性、保守的文本
      例如 0.1 时，生成非常确定的文本，接近贪心解码
    - temperature = 1.0：原始 softmax 概率分布，默认平衡的随机性
    - temperature > 1.0：增加随机性，模型倾向探索低概率 token，生成更多样化、创意的文本
      例如 1.5 时，生成更具创意但可能不太连贯的文本

    数值越小越确定，数值越大越随机。生成创意文本用高温度，生成事实内容用低温度。
    """

    max_tokens: int = 64
    """
    单次生成的最大 token 数（输出序列长度）。

    - 生成将在以下条件之一满足时停止：
      1. 生成了 max_tokens 个 token
      2. 遇到 EOS（End of Sequence）token（如果 ignore_eos=False）
      3. 其他停止条件（如特殊字符、换行等）

    - 值越大，生成的文本越长，但计算成本和延迟越高
    - 建议根据任务设置合理值，例如：
      摘要任务：max_tokens=200
      对话回复：max_tokens=512
      故事生成：max_tokens=2048
    """

    ignore_eos: bool = False
    """
    是否忽略 EOS（End of Sequence）token 来继续生成。

    - False（默认）：遇到 EOS token 就停止生成，这是标准行为
    - True：忽略 EOS token，继续生成直到达到 max_tokens
      用途：某些任务需要强制生成更长的文本，但可能导致质量下降

    EOS token 通常由模型训练时定义，用于标记文本的自然结束点。
    大多数情况下应保持 False。
    """

    def __post_init__(self):
        """
        dataclass 初始化后的参数验证方法。

        确保 temperature 参数满足有效范围，避免配置错误。
        """
        # 验证温度参数：temperature 必须大于非常小的正数（避免 0 或负数）
        # 1e-10 是约等于 0 的极小正数，允许 greedy sampling 的近似但不允许完全贪心
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
   