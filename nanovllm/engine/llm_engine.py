import atexit
from dataclasses import fields
from time import perf_counter

import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class LLMEngine:
    """
    LLM 推理引擎的顶层协调者。

    集成了调度器（Scheduler）和模型运行器（ModelRunner），提供端到端的 LLM 推理服务。
    支持分布式张量并行推理，采用多进程架构提高 GPU 利用率。

    主要职责：
    1. 初始化分布式推理进程
    2. 加载模型和分词器
    3. 管理用户请求队列
    4. 协调调度和模型执行
    5. 整理和返回生成结果
    """

    def __init__(self, model, **kwargs):
        """
        初始化 LLM 推理引擎。

        Args:
            model: 模型名称或路径（HuggingFace 模型卡）
            **kwargs: 配置参数，参考 Config 类中的字段

        流程：
            1. 解析配置参数
            2. 创建分布式进程组（张量并行）
            3. 初始化主进程的模型运行器
            4. 加载分词器并设置 EOS token
            5. 创建调度器
            6. 注册退出处理函数
        """
        # 解析配置参数，过滤出 Config 类中定义的字段
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        # 后端进程列表（用于张量并行）
        self.ps = []
        # 进程间同步事件列表
        self.events = []
        # 获取多进程上下文（spawn 模式确保子进程有完整的环境）
        ctx = mp.get_context("spawn")

        # 创建张量并行的从进程
        for i in range(1, config.tensor_parallel_size):
            # 为每个从进程创建一个同步事件
            event = ctx.Event()
            # 创建从进程，在其中运行 ModelRunner（rank > 0）
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # 主进程初始化模型运行器（rank == 0）
        self.model_runner = ModelRunner(config, 0, self.events)
        # 加载分词器用于 token 编码/解码
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        # 设置 EOS token 用于序列终止判断
        config.eos = self.tokenizer.eos_token_id
        # 初始化调度器管理请求队列
        self.scheduler = Scheduler(config)
        # 注册退出处理函数确保资源清理
        atexit.register(self.exit)

    def exit(self):
        """
        清理资源并优雅退出。

        清理步骤：
        1. 调用模型运行器的 exit 方法清理 GPU 资源
        2. 删除模型运行器引用
        3. 等待所有从进程终止
        """
        # 远程调用模型运行器的 exit 方法（包括所有从进程）
        self.model_runner.call("exit")
        # 删除主进程的模型运行器
        del self.model_runner
        # 等待所有从进程完成
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        添加用户请求到调度队列。

        Args:
            prompt: 输入提示，可以是文本字符串或 token ID 列表
            sampling_params: 采样参数（温度、top_p 等）

        流程：
            1. 将文本提示编码为 token ID（如果需要）
            2. 创建序列对象，绑定采样参数
            3. 加入调度器的等待队列
        """
        # 如果输入是文本字符串，使用分词器编码
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        # 创建序列对象，关联采样参数
        seq = Sequence(prompt, sampling_params)
        # 将序列加入调度器等待队列
        self.scheduler.add(seq)

    def step(self):
        """
        执行一个推理步骤。

        该函数是推理循环的核心，包括调度、执行、后处理三个环节。

        Returns:
            tuple: (outputs, num_tokens)
                - outputs: 完成的序列列表，格式为 [(seq_id, token_ids), ...]
                - num_tokens: 处理的 token 数
                  - Prefill 阶段：返回正数（所有 token 数）
                  - Decode 阶段：返回负数（-序列数）
                  用于计算吞吐量：tokens / time

        流程：
            1. 调度获取待执行的序列（Prefill 或 Decode）
            2. 调用模型运行器执行推理
            3. 后处理更新序列状态（生成的 token 等）
            4. 收集已完成的序列
            5. 计算 token 统计用于吞吐量计算
        """
        # 从调度器获取待执行序列和当前阶段
        seqs, is_prefill = self.scheduler.schedule()
        # 调用模型运行器执行推理，获取生成的 token ID
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 后处理：更新序列的生成 token 和完成状态
        self.scheduler.postprocess(seqs, token_ids)
        # 收集已完成的序列（seq_id 和完成的 token）
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        # 计算 token 数用于吞吐量统计
        # Prefill 时为正数，Decode 时为负数
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        """
        检查所有请求是否已完成。

        Returns:
            bool: 若所有序列都已完成或已清空返回 True，否则返回 False
        """
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        批量生成文本的高级接口。

        这是用户面向的主要 API，提供完整的端到端推理功能，包括请求管理、
        推理循环、进度追踪和结果整理。

        Args:
            prompts: 输入提示列表，可以是文本或 token ID 列表
            sampling_params: 采样参数，可以是单个参数（应用于所有提示）或参数列表
            use_tqdm: 是否显示进度条

        Returns:
            list: 生成结果列表，每个结果是字典格式：
                {
                    "text": 解码后的文本,
                    "token_ids": 生成的 token ID 列表
                }

        流程：
            1. 初始化进度条（可选）
            2. 标准化采样参数到列表格式
            3. 添加所有请求到调度器
            4. 进入推理循环直到所有序列完成
            5. 每步：执行调度 + 推理 + 后处理 + 结果收集
            6. 记录吞吐量统计（Prefill 和 Decode）
            7. 将 token ID 解码为文本
            8. 按原始顺序返回结果
        """
        # 初始化进度条（如果启用）
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        # 标准化采样参数为列表格式
        # 如果传入单个参数，复制为与 prompts 数量相同的列表
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # 添加所有请求到调度器的等待队列
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        # 存储完成序列的结果字典：{seq_id: token_ids}
        outputs = {}
        # 吞吐量统计（用于进度条显示）
        prefill_throughput = decode_throughput = 0.

        # 推理循环：持续执行直到所有序列完成
        while not self.is_finished():
            # 记录步骤开始时间用于吞吐量计算
            t = perf_counter()
            # 执行一个推理步骤
            output, num_tokens = self.step()

            # 更新进度条信息
            if use_tqdm:
                # 根据 num_tokens 符号判断当前阶段
                if num_tokens > 0:
                    # Prefill 阶段：token 数为正
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    # Decode 阶段：token 数为负，需要取反
                    decode_throughput = -num_tokens / (perf_counter() - t)
                # 更新进度条显示吞吐量
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })

            # 收集本步完成的序列
            for seq_id, token_ids in output:
                # 存储该序列的生成 token
                outputs[seq_id] = token_ids
                # 更新进度条（每完成一个序列增加一步）
                if use_tqdm:
                    pbar.update(1)

        # 按原始请求顺序排列结果（seq_id 按升序排列）
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]

        # 将 token ID 列表解码为文本，并格式化为字典
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]

        # 关闭进度条
        if use_tqdm:
            pbar.close()

        return outputs
