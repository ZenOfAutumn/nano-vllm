import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event

import torch
import torch.distributed as dist

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """
    模型运行器，负责在 GPU 上执行模型推理。
    主要职责：
    1. 初始化模型和 KV 缓存
    2. 管理分布式推理（张量并行）
    3. 准备模型输入数据
    4. 执行模型推理和采样
    5. 支持 CUDA Graph 优化
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """
        初始化模型运行器。

        Args:
            config: 配置对象，包含模型、KV 缓存等参数
            rank: 进程排名（用于分布式推理），0 为主进程
            event: 多进程同步事件，rank > 0 时为单个 Event，rank == 0 时为 Event 列表

        流程：
            1. 初始化分布式进程组（NCCL）
            2. 加载模型到 GPU
            3. 分配 KV 缓存
            4. 记录 CUDA Graph（如果启用）
            5. 建立进程间通信（多进程模式）
        """
        # 配置对象
        self.config = config
        hf_config = config.hf_config
        # KV 缓存块大小（tokens 数）
        self.block_size = config.kvcache_block_size
        # 是否强制使用 eager 模式（不使用 CUDA Graph）
        self.enforce_eager = config.enforce_eager
        # 张量并行的世界大小（进程数）
        self.world_size = config.tensor_parallel_size
        # 当前进程的排名
        self.rank = rank
        # 进程间同步事件
        self.event = event

        # 初始化分布式进程组（NCCL 后端）
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        # 设置当前进程使用的 GPU
        torch.cuda.set_device(rank)
        # 保存默认数据类型
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        # 初始化模型
        self.model = Qwen3ForCausalLM(hf_config)
        # 加载预训练权重
        load_model(self.model, config.model)
        # 初始化采样器
        self.sampler = Sampler()
        # 热启动模型（预热 GPU 缓存）
        self.warmup_model()
        # 分配 KV 缓存
        self.allocate_kv_cache()
        # 记录 CUDA Graph（如果不使用 eager 模式）
        if not self.enforce_eager:
            self.capture_cudagraph()
        # 恢复默认设备和数据类型
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 多进程模式：建立进程间共享内存和通信
        if self.world_size > 1:
            if rank == 0:
                # 主进程：创建共享内存
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                # 从进程：连接到共享内存并开启事件循环
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        """
        清理资源并退出模型运行器。

        清理内容：
        1. 关闭共享内存
        2. 删除 CUDA Graph 对象
        3. 同步 GPU 操作
        4. 销毁分布式进程组
        """
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """
        从进程事件循环（仅用于非主进程）。

        无限循环读取共享内存中的方法调用，执行后同步。
        当收到 "exit" 方法时退出循环。
        """
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """
        从共享内存读取方法调用（仅从进程）。

        Returns:
            tuple: (method_name, args) - 方法名和参数列表
        """
        assert self.world_size > 1 and self.rank > 0
        # 等待主进程写入数据的信号
        self.event.wait()
        # 读取数据长度（前 4 字节）
        n = int.from_bytes(self.shm.buf[0:4], "little")
        # 反序列化方法名和参数
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        # 清除事件信号
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """
        写入方法调用到共享内存（仅主进程）。

        Args:
            method_name: 要调用的方法名
            *args: 方法的参数
        """
        assert self.world_size > 1 and self.rank == 0
        # 序列化方法名和参数
        data = pickle.dumps([method_name, *args])
        n = len(data)
        # 写入数据长度（前 4 字节）
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        # 写入序列化数据
        self.shm.buf[4:n+4] = data
        # 向所有从进程发送信号
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """
        调用指定方法，支持分布式同步。

        在多进程模式中，主进程会向共享内存写入方法调用，从进程会从事件循环中读取。

        Args:
            method_name: 要调用的方法名
            *args: 方法的参数

        Returns:
            方法的返回值
        """
        # 多进程模式：主进程写入共享内存，从进程会从 loop 中读取
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        # 执行方法
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """
        热启动模型。

        运行一次前向传播来预热 GPU 内存和缓存。
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # 计算热启动的序列数
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        # 创建虚拟序列（全为 0 token）
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        # 执行一次前向传播
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        分配 KV 缓存。

        根据 GPU 内存计算能分配的最大块数，创建连续的 KV 缓存张量，
        并将其绑定到模型的各个注意力层。
        """
        config = self.config
        hf_config = config.hf_config
        # 查询 GPU 内存信息
        free, total = torch.cuda.mem_get_info()
        used = total - free
        # 查询当前内存使用统计
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # 计算分布式情况下每个进程的 KV 头数
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        # 获取注意力头维度
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 计算单个块的字节数：2 (K + V) * 层数 * 块大小 * 头数 * 头维度 * 数据类型字节数
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        # 计算能分配的最大块数
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        # 创建 KV 缓存张量：(K/V, 层数, 块数, 块大小, 头数, 头维度)
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        # 将 KV 缓存分配给模型的各个注意力层
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """
        准备块表张量。

        将序列的块表对齐到相同长度（补充 -1），转换为 GPU 张量。

        Args:
            seqs: 序列列表

        Returns:
            torch.Tensor: 形状为 (序列数, 最大块数) 的块表张量
        """
        # 找到最长的块表长度
        max_len = max(len(seq.block_table) for seq in seqs)
        # 对齐所有块表，用 -1 填充不足的部分
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        # 转换为 GPU 张量
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        准备 Prefill 阶段的输入数据。

        将序列数据转换为模型所需的格式：
        - input_ids: 未缓存的 token ID
        - positions: 对应的位置索引
        - cu_seqlens_q/k: 查询/键的累积长度（用于 flash attention）
        - slot_mapping: KV 缓存中的槽位映射

        Args:
            seqs: 序列列表

        Returns:
            tuple: (input_ids, positions) - 模型输入张量
        """
        input_ids = []
        positions = []
        # 累积序列长度（用于 Flash Attention 的分组）
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        # KV 缓存槽位映射
        slot_mapping = []
        block_tables = None

        for seq in seqs:
            seqlen = len(seq)
            # 只提取未缓存的 token
            input_ids.extend(seq[seq.num_cached_tokens:])
            # 对应的位置信息
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            # 查询长度（只有新 token）和键长度（全部 token）
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            # 累积长度用于 Flash Attention 分组
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            # 跳过热启动序列（无块表）
            if not seq.block_table:    # warmup
                continue
            # 为新 token 分配 KV 缓存槽位
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    # 最后一个块可能不满
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))

        # 如果有前缀缓存，需要块表
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)

        # 转换为 GPU 张量
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 设置上下文信息供模型使用
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        准备 Decode 阶段的输入数据。

        在 decode 阶段，每个序列只生成一个新 token，输入为最后一个 token，
        位置为序列长度减 1。

        Args:
            seqs: 序列列表

        Returns:
            tuple: (input_ids, positions) - 模型输入张量
        """
        input_ids = []
        positions = []
        # KV 缓存槽位映射（每个序列一个新槽位）
        slot_mapping = []
        # 上下文长度（序列已有的 token 数）
        context_lens = []

        for seq in seqs:
            # 只输入最后一个 token
            input_ids.append(seq.last_token)
            # 位置是序列长度减 1
            positions.append(len(seq) - 1)
            # 上下文长度就是序列长度
            context_lens.append(len(seq))
            # KV 缓存槽位：最后一块中的最后一个 token
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)

        # 转换为 GPU 张量
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        # 设置上下文信息供模型使用
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """
        准备采样参数。

        提取每个序列的温度参数用于 token 采样。

        Args:
            seqs: 序列列表

        Returns:
            torch.Tensor: 形状为 (序列数,) 的温度张量
        """
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        执行模型前向传播。

        在 prefill 阶段或序列长度较长时使用 eager 模式（动态执行）；
        在 decode 阶段和序列较短时使用 CUDA Graph 优化。

        Args:
            input_ids: 模型输入的 token ID 张量
            positions: 对应的位置张量
            is_prefill: 是否为 prefill 阶段

        Returns:
            torch.Tensor: 模型输出的 logits
        """
        # 在 prefill 阶段、eager 模式或批大小过大时，直接执行（不使用 CUDA Graph）
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # Decode 阶段使用 CUDA Graph 进行优化
            bs = input_ids.size(0)
            context = get_context()
            # 选择合适的 CUDA Graph（找到第一个 >= bs 的大小）
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            # 更新 graph 变量
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # 重播 CUDA Graph
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        执行一完整的推理步骤：准备数据、运行模型、采样。

        Args:
            seqs: 序列列表
            is_prefill: 是否为 prefill 阶段

        Returns:
            list[int]: 采样得到的 token ID 列表（仅主进程返回有效值，从进程返回 None）
        """
        # 根据阶段准备模型输入
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # 准备采样参数（仅主进程需要）
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        # 运行模型得到 logits
        logits = self.run_model(input_ids, positions, is_prefill)
        # 采样生成 token ID（仅主进程）
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        # 重置上下文
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        记录 CUDA Graph 用于 decode 阶段的优化。

        对不同的批大小（1, 2, 4, 8, 16, 32, ...）预先记录模型执行图，
        这样在 decode 阶段可以直接重播图而不需要动态编译，大幅提升性能。
        """
        config = self.config
        hf_config = config.hf_config
        # 最大批大小（取决于最大并发序列数和 GPU 限制）
        max_bs = min(self.config.max_num_seqs, 512)
        # 计算最大块表长度
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        # 初始化 CUDA Graph 变量张量
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        # 要记录的批大小列表：1, 2, 4, 8, 16, 32, ..., max_bs
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 倒序记录图（先记录大的批大小能复用内存池）
        for bs in reversed(self.graph_bs):
            # 创建新的 CUDA Graph
            graph = torch.cuda.CUDAGraph()
            # 设置上下文
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            # 热启动（让 GPU 预热）
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            # 开始记录
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            # 第一个图创建内存池
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            # 保存图
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 保存图变量以便后续重播时更新
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
