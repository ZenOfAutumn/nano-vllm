from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """序列的状态枚举"""
    WAITING = auto()
    """等待状态：序列已创建但尚未开始处理"""
    RUNNING = auto()
    """运行状态：序列正在生成中"""
    FINISHED = auto()
    """完成状态：序列生成已完成或已停止"""


class Sequence:
    """
    LLM 推理中的单个请求序列类。

    代表从用户到模型的一个请求，包含 prompt token 和生成的 completion token。
    管理序列的生命周期、缓存状态和采样参数。
    """

    block_size = 256
    """KV 缓存块大小（固定为 256），用于分块管理缓存"""

    counter = count()
    """全局序列 ID 计数器，为每个序列分配唯一的 ID"""

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        """
        初始化一个新的序列对象。

        参数：
            token_ids (list[int]): 初始的 prompt token ID 列表
                例如：[1, 2, 3] 表示输入的三个 token

            sampling_params (SamplingParams): 采样参数配置，默认为 SamplingParams()
                包含 temperature、max_tokens、ignore_eos 等生成参数

        初始化的属性：
            seq_id: 序列的全局唯一 ID（从 counter 自增获得）
            status: 初始状态为 WAITING（等待处理）
            token_ids: 所有 token 的列表（prompt + completion）
            last_token: 最后一个 token ID（初始为 prompt 的最后一个）
            num_tokens: 当前总 token 数（初始等于 prompt_tokens）
            num_prompt_tokens: prompt token 的固定数量（不会改变）
            num_cached_tokens: 已缓存的 token 数（0 表示未缓存）
            block_table: 块表，用于追踪 KV 缓存块的分配情况
            temperature: 采样温度参数
            max_tokens: 最大生成 token 数
            ignore_eos: 是否忽略 EOS token
        """
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        """
        返回序列的总长度（包括 prompt 和已生成的 completion）。

        返回值：
            int: 序列中的总 token 数

        使用例：
            seq = Sequence([1, 2, 3])
            len(seq)  # 返回 3
        """
        return self.num_tokens

    def __getitem__(self, key):
        """
        支持按索引访问序列中的 token。

        参数：
            key (int): 要访问的 token 在列表中的索引

        返回值：
            int: 索引位置的 token ID

        使用例：
            seq = Sequence([1, 2, 3])
            seq[0]  # 返回 1
            seq[-1]  # 返回 3（最后一个 token）
        """
        return self.token_ids[key]

    @property
    def is_finished(self):
        """
        检查序列是否已完成生成。

        返回值：
            bool: True 表示序列状态为 FINISHED，False 表示还在等待或运行

        使用例：
            if seq.is_finished:
                print("生成完成")
        """
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """
        计算已生成的 completion token 数量。

        返回值：
            int: completion token 的数量 = 总 token 数 - prompt token 数

        使用例：
            seq = Sequence([1, 2, 3])  # 3 个 prompt tokens
            seq.append_token(4)
            seq.append_token(5)
            seq.num_completion_tokens  # 返回 2（生成了 2 个 token）
        """
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """
        获取序列中的 prompt token ID 列表。

        返回值：
            list[int]: prompt 部分的 token ID 列表

        注意：
            prompt 部分的长度是固定的，不会随着生成而改变

        使用例：
            seq = Sequence([1, 2, 3])
            seq.append_token(4)
            seq.prompt_token_ids  # 返回 [1, 2, 3]
        """
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """
        获取序列中的 completion（生成的）token ID 列表。

        返回值：
            list[int]: 模型生成部分的 token ID 列表

        使用例：
            seq = Sequence([1, 2, 3])
            seq.append_token(4)
            seq.append_token(5)
            seq.completion_token_ids  # 返回 [4, 5]
        """
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        """
        计算已缓存的 KV 缓存块数量。

        返回值：
            int: 缓存块数 = 已缓存的 token 数 // block_size

        说明：
            KV 缓存按块（block）管理以提高效率。
            每个块包含 256 个 token 的 key-value 对。

        使用例：
            seq.num_cached_tokens = 512
            seq.num_cached_blocks  # 返回 2（512 // 256 = 2）
        """
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """
        计算序列所需的总块数（包括不完整的块）。

        返回值：
            int: 所需块数 = ceil(总 token 数 / block_size)

        说明：
            使用向上取整公式：(num_tokens + block_size - 1) // block_size
            确保即使最后一块不完整也会被计算

        使用例：
            seq.num_tokens = 300
            seq.num_blocks  # 返回 2（300 / 256 向上取整 = 2）

            seq.num_tokens = 512
            seq.num_blocks  # 返回 2（512 / 256 = 2）

            seq.num_tokens = 513
            seq.num_blocks  # 返回 3（513 / 256 向上取整 = 3）
        """
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """
        计算最后一个块中包含的 token 数量。

        返回值：
            int: 最后一块中的 token 数

        说明：
            最后一块可能不满（不是 256 个 token）
            计算公式：总 token 数 - (块数 - 1) * block_size

        使用例：
            seq.num_tokens = 300  （2 块）
            seq.last_block_num_tokens  # 返回 44（300 - 1*256 = 44）

            seq.num_tokens = 512  （2 块）
            seq.last_block_num_tokens  # 返回 256（512 - 1*256 = 256）
        """
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """
        获取序列中第 i 个块的 token ID 列表。

        参数：
            i (int): 块的索引（0 开始）

        返回值：
            list[int]: 第 i 个块中的所有 token ID

        说明：
            每个块最多包含 256 个 token（最后一块可能更少）
            使用断言检查索引有效性

        使用例：
            seq = Sequence([1, 2, ..., 300])
            seq.block(0)  # 返回 [1, 2, ..., 256]
            seq.block(1)  # 返回 [257, 258, ..., 300]
            seq.block(2)  # 抛出 AssertionError（超出范围）
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """
        向序列追加一个新的生成 token。

        参数：
            token_id (int): 要添加的 token ID

        说明：
            这个方法在模型推理时每生成一个 token 就调用一次
            会更新：token_ids 列表、last_token、num_tokens

        使用例：
            seq = Sequence([1, 2, 3])
            seq.append_token(4)
            seq.token_ids  # [1, 2, 3, 4]
            seq.last_token  # 4
            seq.num_tokens  # 4
            seq.num_completion_tokens  # 1（只有 1 个生成的 token）
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """
        用于序列化（pickle）序列对象的方法。

        返回值：
            tuple: 包含序列状态的元组，用于保存/恢复

        说明：
            这是 Python 魔术方法，支持 pickle 序列化
            优化存储：如果没有生成 token（只有 prompt），保存整个 token_ids
                     如果已生成 token，只保存最后一个 token（节省空间）
            返回格式：(num_tokens, num_prompt_tokens, num_cached_tokens,
                     block_table, token_ids 或 last_token)

        设计意图：
            - 减少内存使用，因为 prompt_token_ids 在序列化后不会改变
            - 只需保存最后一个生成的 token 就能恢复完整信息
        """
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        """
        用于反序列化（unpickle）序列对象的方法。

        参数：
            state (tuple): 由 __getstate__ 返回的状态元组

        说明：
            这是 Python 魔术方法，与 __getstate__ 配对使用
            恢复逻辑：
            - 如果没有生成 token（num_completion_tokens == 0），
              state[-1] 是完整的 token_ids 列表
            - 如果已生成 token，
              state[-1] 是最后生成的 token，需要单独处理

        使用例：
            # 序列化
            state = seq.__getstate__()

            # 反序列化
            new_seq = Sequence([])
            new_seq.__setstate__(state)
        """
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
