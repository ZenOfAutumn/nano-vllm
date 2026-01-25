from collections import deque

import numpy as np
import xxhash

from nanovllm.engine.sequence import Sequence


class Block:
    """
    KV 缓存块（Block）类。

    代表 KV 缓存中的一个固定大小的块（如 256 个 token）。
    每个块存储该段 token 的 key-value 对，支持重复使用和共享。
    """

    def __init__(self, block_id):
        """
        初始化块对象。

        参数：
            block_id (int): 块的唯一标识符（从 0 开始）
        """
        self.block_id = block_id
        """块的唯一 ID"""

        self.ref_count = 0
        """
        引用计数：有多少个序列正在使用这个块。

        - ref_count = 0：块空闲，可以被分配或释放
        - ref_count >= 1：块被使用中，不能释放
        - 支持块共享：多个序列可以共享同一块（如果内容相同）
        """

        self.hash = -1
        """
        块内容的哈希值（用于块复用检测）。

        - hash = -1：未计算哈希，通常是不完整的块
        - hash >= 0：已计算哈希值，可以用来检测重复块

        应用：如果两个块的哈希值相同且内容相同，可以复用同一块。
        """

        self.token_ids = []
        """块中包含的 token ID 列表，大小通常是 256"""

    def update(self, hash: int, token_ids: list[int]):
        """
        更新块的内容和哈希值。

        参数：
            hash (int): 块内容的哈希值
            token_ids (list[int]): 块中的 token ID 列表

        说明：
            当块被填满（达到 block_size）时调用此方法，
            以保存块的最终内容和哈希值。
        """
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """
        重置块为初始状态。

        说明：
            当块被重新分配时调用，清空之前的内容。
            将 ref_count 设为 1 表示块被分配给一个序列。
        """
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """
    块管理器（Block Manager）类。

    管理 KV 缓存的分配和回收。支持块的复用、共享和高效的显存管理。

    核心功能：
    1. 块分配：为新序列分配块
    2. 块复用：检测并复用相同内容的块
    3. 块共享：多个序列可以共享同一块（提高显存利用率）
    4. 块回收：释放不再使用的块
    """

    def __init__(self, num_blocks: int, block_size: int):
        """
        初始化块管理器。

        参数：
            num_blocks (int): 总块数（决定最大缓存容量）
                例如：num_blocks=100, block_size=256 → 最多缓存 25600 个 token

            block_size (int): 每个块的大小（token 数）
                通常为 256（与 Sequence.block_size 保持一致）
        """
        self.block_size = block_size
        """每个块的固定大小（256 个 token）"""

        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        """
        所有块的列表（容量固定）。

        索引：block_id 直接映射到列表索引
        状态：空闲或已使用，通过 ref_count 判断
        """

        self.hash_to_block_id: dict[int, int] = dict()
        """
        哈希值到块 ID 的映射表（用于块复用）。

        作用：快速查找是否存在相同内容的块
        例如：hash_to_block_id[12345] = 5（哈希值 12345 对应块 5）

        应用场景：
        - 多个序列有相同的 prompt → 复用同一块
        - 计算哈希值，查找映射表，避免重复存储
        """

        self.free_block_ids: deque[int] = deque(range(num_blocks))
        """
        空闲块的队列。

        - 新块：从 free_block_ids 中取出
        - 块释放：被放回 free_block_ids 队尾
        - 队列特性：FIFO 分配，局部性好
        """

        self.used_block_ids: set[int] = set()
        """
        正在使用中的块的集合。

        - 块分配时添加到此集合
        - 块释放时从此集合删除
        - O(1) 查询：快速检查块是否被使用
        """

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算 token 序列的哈希值（用于块复用）。

        参数：
            token_ids (list[int]): 要计算哈希的 token 列表

            prefix (int): 前缀哈希值（默认 -1）
                - -1：不使用前缀，计算单块哈希
                - >=0：前一块的哈希，用于计算链式哈希

        返回值：
            int: 计算得到的 xxhash64 哈希值

        说明：
            使用 xxhash 库进行快速哈希计算。
            支持前缀哈希，可以链式计算多块的哈希。

        应用：
            block0 = compute_hash([1, 2, 3, 4], -1)        # 计算块 0
            block1 = compute_hash([5, 6, 7, 8], block0)    # 用块 0 的哈希作前缀
            这样可以检测两块的组合是否在缓存中存在。
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """
        内部方法：分配一个块。

        参数：
            block_id (int): 要分配的块 ID（从 free_block_ids 中选出）

        返回值：
            Block: 分配后的块对象

        说明：
            - 块必须是空闲的（ref_count = 0）
            - 从 free_block_ids 中移除
            - 添加到 used_block_ids 中
            - 重置块的内部状态
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """
        内部方法：回收一个块。

        参数：
            block_id (int): 要回收的块 ID

        说明：
            - 块的引用计数必须为 0
            - 从 used_block_ids 中移除
            - 添加回 free_block_ids 队尾（FIFO）
        """
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """
        检查是否有足够的空闲块来分配给序列。

        参数：
            seq (Sequence): 要分配的序列

        返回值：
            bool: True 表示有足够块，False 表示块不足

        说明：
            用于预检查，避免分配失败导致推理中断。
            在正式调用 allocate() 前应该先调用此方法。
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        为序列分配 KV 缓存块。

        参数：
            seq (Sequence): 要分配块的序列

        说明：
            核心功能：遍历序列的每一块，尝试复用（通过哈希检测）。

        流程：
            1. 遍历序列的每个块
            2. 计算块的哈希值（检测是否可以复用）
            3. 查找哈希表：是否已有相同内容的块
            4. 如果找到且内容相同 → 复用（ref_count++，num_cached_tokens++）
            5. 如果没找到 → 分配新块
            6. 将块 ID 添加到 seq.block_table

        优化：
            - 块复用：相同内容只保存一份（如 prompt 共享）
            - 块共享：多个序列共享同一块（ref_count 管理）
            - 缓存统计：num_cached_tokens 记录已缓存的 token 数
        """
        # 前提条件：序列的块表必须是空的（尚未分配过）
        assert not seq.block_table

        # h：当前块的哈希值（用于链式哈希计算）
        # 初始化为 -1 表示还没计算哈希
        h = -1

        # cache_miss：标志位，表示是否发生了缓存未命中
        # False 表示所有块都可以从缓存复用
        # True 表示当前块及之后的块都需要分配新内存
        cache_miss = False

        # 遍历序列的每个块（块大小通常为 256 个 token）
        for i in range(seq.num_blocks):
            # 获取第 i 个块的所有 token ID
            token_ids = seq.block(i)

            # 计算哈希值（用于快速检测块是否可复用）
            # 只有当块满了（len == block_size）才计算哈希
            # 不完整的块（最后一块）哈希值设为 -1
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1

            # 查找哈希表：是否已有相同内容的块
            # hash_to_block_id 是一个字典，存储 哈希值 → 块ID 的映射
            # 如果找到，返回块 ID；如果没找到，返回 -1
            block_id = self.hash_to_block_id.get(h, -1)

            # 检查是否可以复用这个块
            # 两个条件之一满足就表示不能复用：
            # 1. block_id == -1：哈希表中没有这个哈希值
            # 2. 哈希相同但内容不同：防止哈希碰撞（极少但可能发生）
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                # 不能复用 → 发生缓存未命中
                cache_miss = True

            # 根据是否发生缓存未命中来处理块分配
            if cache_miss:
                # 缓存未命中：需要分配新块
                # 从空闲块队列的队头取出一个块
                block_id = self.free_block_ids[0]
                # 调用内部方法正式分配这个块
                # （移出空闲队列，加入使用队列，重置块状态）
                block = self._allocate_block(block_id)
            else:
                # 缓存命中：可以复用现有的块
                # 增加已缓存的 token 数（统计信息）
                seq.num_cached_tokens += self.block_size

                # 检查这个块是否已在使用中
                if block_id in self.used_block_ids:
                    # 块已被其他序列使用 → 实现块共享
                    # 获取块对象
                    block = self.blocks[block_id]
                    # 增加引用计数（表示又有一个序列在使用这个块）
                    block.ref_count += 1
                else:
                    # 块在哈希表中但还没被使用（之前分配但已回收）
                    # 需要重新分配这个块
                    block = self._allocate_block(block_id)

            # 如果哈希值有效（不是 -1），更新块的内容和哈希表
            if h != -1:
                # 更新块的哈希值和 token 内容（块填满时调用）
                block.update(h, token_ids)
                # 更新哈希表的映射关系，以便后续查找
                self.hash_to_block_id[h] = block_id

            # 将块 ID 添加到序列的块表中
            # block_table 记录了序列所有块的 ID 列表
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """
        回收序列占用的所有块。

        参数：
            seq (Sequence): 要回收的序列

        说明：
            当序列完成生成或被中止时调用。

        流程：
            1. 逆序遍历 block_table（从后往前）
            2. 每个块的 ref_count 减 1
            3. 如果 ref_count 变为 0，调用 _deallocate_block 回收
            4. 重置 seq 的缓存统计和块表

        为什么逆序？
            - 按生成顺序反向释放，便于内存管理和调试
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """
        检查是否有足够的空闲块来追加新 token。

        参数：
            seq (Sequence): 要追加 token 的序列

        返回值：
            bool: True 表示可以追加，False 表示需要等待

        说明：
            当序列长度 % block_size == 1 时（刚好填满一块），
            需要为下一个 token 预分配一个新块。

        例子：
            - block_size = 256
            - 当序列长度为 257 时（1 个完整块 + 1 个新 token）
            - 需要 1 个空闲块来存放第 2 个块
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        处理序列追加新 token 时的块管理。

        参数：
            seq (Sequence): 正在追加 token 的序列

        说明：
            调用时机：每生成一个新 token 后，需要更新块状态。

        处理三种情况：

        情况 1：len(seq) % block_size == 1（刚好完成一块，开始新块）
            - 最后一块已完成，哈希值有效
            - 分配一个新块来存放第一个新 token

        情况 2：len(seq) % block_size == 0（刚好填满一块）
            - 最后一块已满，计算其哈希值
            - 考虑前缀哈希（与前一块的关系）
            - 更新哈希表，支持这个块的复用

        情况 3：其他（块还未填满）
            - 最后一块还在继续增长
            - 不需要特殊处理

        优化：
            - 支持链式哈希计算（考虑块的前后关系）
            - 及时更新哈希表，为后续块复用做准备
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            # 情况 1：刚完成一块，分配新块
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            # 情况 2：刚填满一块，计算并记录哈希
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            # 情况 3：块还未填满，无需操作
            assert last_block.hash == -1
