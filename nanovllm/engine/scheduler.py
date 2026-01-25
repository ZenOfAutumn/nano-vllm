from collections import deque

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    """
    调度器，负责管理序列的执行流程，包括等待队列、运行队列和内存分配。
    主要职责：
    1. 维护等待和运行的序列队列
    2. 调度序列的执行（prefill 和 decode 阶段）
    3. 处理序列的抢占和恢复
    4. 后处理生成的 token
    """

    def __init__(self, config: Config):
        """
        初始化调度器。

        Args:
            config: 配置对象，包含调度器和内存管理的相关参数
        """
        # 最大并发序列数
        self.max_num_seqs = config.max_num_seqs
        # 单次批处理的最大 token 数
        self.max_num_batched_tokens = config.max_num_batched_tokens
        # 序列结束标记的 token ID
        self.eos = config.eos
        # KV 缓存块管理器
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # 等待调度的序列队列
        self.waiting: deque[Sequence] = deque()
        # 正在运行的序列队列
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """
        检查所有序列是否都已完成。

        Returns:
            bool: 如果等待队列和运行队列都为空，则返回 True
        """
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """
        添加新的序列到等待队列。

        Args:
            seq: 待添加的序列对象
        """
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        调度序列执行。包含两个阶段：
        1. prefill 阶段：将等待序列加入运行队列，分配必要的 KV 缓存
        2. decode 阶段：从运行队列中选择序列继续执行

        Returns:
            tuple[list[Sequence], bool]: 返回调度的序列列表和是否为 prefill 阶段的标记
                - True: prefill 阶段
                - False: decode 阶段
        """
        # prefill 阶段：从等待队列中选择序列并分配内存
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0

        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            # 检查是否超过 token 数限制或内存分配失败
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            # 为序列分配 KV 缓存块
            self.block_manager.allocate(seq)
            # 累计 token 数（只计算未缓存的 token）
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            # 更新序列状态为运行中
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode 阶段：从运行队列中选择序列继续执行
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            # 检查是否有足够空间追加新 token
            while not self.block_manager.can_append(seq):
                # 如果空间不足，尝试抢占其他序列
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    # 如果没有其他序列可抢占，则抢占当前序列
                    self.preempt(seq)
                    break
            else:
                # 成功追加 token 空间
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        assert scheduled_seqs
        # 恢复序列到运行队列（保持原顺序）
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        抢占（暂停）一个正在运行的序列，将其返回到等待队列。
        用于释放内存以支持其他序列的执行。

        Args:
            seq: 待抢占的序列对象
        """
        # 更新序列状态为等待中
        seq.status = SequenceStatus.WAITING
        # 释放该序列占用的 KV 缓存块
        self.block_manager.deallocate(seq)
        # 将序列添加到等待队列的前端（优先调度）
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """
        后处理阶段，更新序列的生成 token，检查序列是否完成。

        Args:
            seqs: 处理的序列列表
            token_ids: 模型生成的 token ID 列表，与序列列表一一对应

        Returns:
            list[bool]: 每个序列是否完成的标记列表（此实现未使用返回值）
        """
        for seq, token_id in zip(seqs, token_ids):
            # 将生成的 token 添加到序列
            seq.append_token(token_id)
            # 检查序列是否应该停止：遇到 EOS token 或达到最大长度
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                # 标记序列为已完成
                seq.status = SequenceStatus.FINISHED
                # 释放该序列占用的 KV 缓存块
                self.block_manager.deallocate(seq)
                # 从运行队列中移除该序列
                self.running.remove(seq)
