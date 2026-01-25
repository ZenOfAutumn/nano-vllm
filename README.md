<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<h1 align="center">Nano-vLLM</h1>

<p align="center">
一个从零开始构建的轻量级 vLLM 实现，专注于高性能离线推理。
</p>

## 📋 目录

- [简介](#简介)
- [核心特性](#核心特性)
- [架构设计](#架构设计)
- [安装指南](#安装指南)
- [模型下载](#模型下载)
- [快速开始](#快速开始)
- [API 文档](#api-文档)
- [性能基准](#性能基准)
- [代码结构](#代码结构)
- [优化技术](#优化技术)
- [常见问题](#常见问题)
- [许可证](#许可证)

## 简介

Nano-vLLM 是一个从头实现的轻量级 vLLM 框架，设计用于在 GPU 上进行高效的离线推理。相比 vLLM 的完整实现，Nano-vLLM 提供了一个更加精简的代码库（约 1,200 行 Python 代码），同时保持了可比的推理速度。该项目的目标是为开发者提供一个易于理解、易于扩展的大模型推理框架。

### 主要优势

- **可读性强**：代码精简清晰，易于理解和学习
- **性能优异**：推理吞吐量与 vLLM 相当，在某些配置下更优
- **功能完整**：支持动态批处理、张量并行、内存优化等高级特性
- **易于部署**：快速集成，支持 Hugging Face 模型生态

## 核心特性

* 🚀 **高速离线推理** - 推理速度可与 vLLM 媲美
* 📖 **代码可读性强** - 精简的 Python 实现，易于学习
* ⚡ **完整优化套件** - 前缀缓存、张量并行、Torch 编译、CUDA 图等
* 🔄 **动态批处理** - 智能调度多个推理任务
* 💾 **内存高效** - KV 缓存块管理和智能预取策略
* 🎯 **张量并行** - 支持多 GPU 分布式推理

## 架构设计

### 整体架构

Nano-vLLM 的核心架构分为三个主要层次：

```
┌─────────────────────────────────────┐
│        LLM (用户接口层)              │
│    - generate() 推理入口方法         │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│     LLMEngine (引擎层)               │
│    - 请求管理                       │
│    - 调度控制                       │
│    - Token 解码                     │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│    Scheduler (调度器)                │
│    - 前填充/解码调度                │
│    - 序列抢占管理                   │
│    - KV 缓存块分配                  │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│   ModelRunner (模型执行层)           │
│    - 模型推理执行                   │
│    - CUDA 图优化                    │
│    - 张量并行通信                   │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│   Model & Layers (模型层)            │
│    - Qwen3ForCausalLM               │
│    - 注意力、MLP、嵌入等             │
└─────────────────────────────────────┘
```

### 关键组件

#### 1. 序列管理 (Sequence)
- **作用**：管理推理过程中的单个请求序列
- **核心属性**：
  - `token_ids`：完整的 token ID 列表
  - `block_table`：KV 缓存块表映射
  - `status`：序列状态（等待/运行/完成）
  - `num_cached_tokens`：已缓存的 token 数量

#### 2. 块管理器 (BlockManager)
- **作用**：管理 KV 缓存块的分配和回收，实现前缀缓存
- **核心功能**：
  - 块级别的内存管理
  - 基于 hash 的前缀缓存检测
  - 内存高效的块复用

#### 3. 调度器 (Scheduler)
- **作用**：负责请求的智能调度和资源分配
- **调度流程**：
  - **前填充阶段**：处理提示词 tokens，生成 KV 缓存
  - **解码阶段**：逐 token 生成输出
  - **抢占机制**：当资源不足时抢占低优先级序列

#### 4. 模型执行器 (ModelRunner)
- **作用**：执行模型推理和采样
- **核心功能**：
  - 模型前向传播
  - CUDA 图捕获和重放
  - 张量并行通信
  - 模型预热和 KV 缓存分配

## 安装指南

### 系统要求

- Python >= 3.10, < 3.13
- CUDA >= 11.8（建议）
- GPU 内存：至少 8GB（用于推理）

### 安装步骤

```bash
# 从 GitHub 安装
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git

# 或克隆后本地安装
git clone https://github.com/GeeeekExplorer/nano-vllm.git
cd nano-vllm
pip install -e .
```

### 依赖项

核心依赖包括：
- **torch >= 2.4.0** - PyTorch 深度学习框架
- **transformers >= 4.51.0** - 模型加载和分词器
- **triton >= 3.0.0** - 自定义 CUDA 内核编写
- **flash-attn** - 高效注意力实现
- **xxhash** - 快速哈希用于前缀缓存

## 模型下载

### 使用 Hugging Face CLI

```bash
# 下载 Qwen3-0.6B 模型
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

### 下载其他模型

Nano-vLLM 支持任何符合 Hugging Face 标准的模型，只要具备：
- 标准的 `config.json`
- `.safetensors` 格式的权重文件
- 兼容的架构（建议使用类似 Qwen、LLaMA 的模型）

## 快速开始

### 基础示例

```python
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

# 初始化模型
model_path = "/path/to/model"
llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)

# 定义采样参数
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

# 推理
prompts = ["Hello, Nano-vLLM.", "What is AI?"]
outputs = llm.generate(prompts, sampling_params)

# 获取结果
for output in outputs:
    print(output["text"])
    print("Token IDs:", output["token_ids"])
```

### 使用聊天模板

```python
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_path = "/path/to/model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)

# 使用模型的聊天模板
prompt = "Introduce yourself"
formatted_prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate([formatted_prompt], sampling_params)
print(outputs[0]["text"])
```

### 批量推理

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/model")

# 支持单一采样参数
prompts = ["问题1", "问题2", "问题3"]
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(prompts, sampling_params)

# 或者为每个提示指定不同的采样参数
sampling_params_list = [
    SamplingParams(temperature=0.6, max_tokens=256),
    SamplingParams(temperature=0.8, max_tokens=512),
    SamplingParams(temperature=0.5, max_tokens=128),
]
outputs = llm.generate(prompts, sampling_params_list)
```

### 张量并行推理

```python
from nanovllm import LLM, SamplingParams

# 使用 2 个 GPU 进行张量并行推理
llm = LLM(
    "/path/to/model",
    tensor_parallel_size=2,
    max_num_seqs=256,
    gpu_memory_utilization=0.9
)

prompts = ["Question 1", "Question 2"]
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(prompts, sampling_params)
```

## API 文档

### LLM 类

主要的推理接口类。

#### 初始化参数

```python
llm = LLM(
    model,                      # str: 模型路径
    max_num_batched_tokens=16384,  # int: 单个批次最大 token 数
    max_num_seqs=512,           # int: 单个批次最大序列数
    max_model_len=4096,         # int: 模型最大序列长度
    gpu_memory_utilization=0.9, # float: GPU 内存利用率
    tensor_parallel_size=1,     # int: 张量并行 GPU 数量
    enforce_eager=False,        # bool: 禁用 CUDA 图优化
    kvcache_block_size=256,     # int: KV 缓存块大小
)
```

#### 主要方法

##### generate()

```python
outputs = llm.generate(
    prompts,              # list[str] | list[list[int]]: 输入提示或 token ID
    sampling_params,      # SamplingParams | list[SamplingParams]: 采样参数
    use_tqdm=True        # bool: 是否显示进度条
) -> list[dict]
```

**返回值**：列表，每个元素为字典，包含：
- `text`：生成的文本
- `token_ids`：生成的 token ID 列表

### SamplingParams 类

采样参数配置。

#### 参数说明

```python
sampling_params = SamplingParams(
    temperature=1.0,    # float: 采样温度（>1e-10）
    max_tokens=64,      # int: 最大生成 token 数
    ignore_eos=False,   # bool: 是否忽略 EOS token
)
```

**参数详解**：
- **temperature**：控制采样的随机性，值越小越确定，越大越随机
- **max_tokens**：生成序列的最大长度（包括提示词）
- **ignore_eos**：如果为 False，遇到 EOS token 就停止生成

### Config 类

内部配置类，用于初始化引擎。

```python
@dataclass
class Config:
    model: str                           # 模型路径
    max_num_batched_tokens: int = 16384  # 批处理 token 上限
    max_num_seqs: int = 512              # 批处理序列上限
    max_model_len: int = 4096            # 模型序列长度上限
    gpu_memory_utilization: float = 0.9  # GPU 内存利用率
    tensor_parallel_size: int = 1        # TP 大小
    enforce_eager: bool = False          # 是否禁用 CUDA 图
    kvcache_block_size: int = 256        # 块大小
```

## 性能基准

### 测试配置

- **硬件**：RTX 4070 Laptop (8GB 显存)
- **模型**：Qwen3-0.6B
- **批处理**：256 个序列
- **输入长度**：100–1024 tokens（随机采样）
- **输出长度**：100–1024 tokens（随机采样）

### 性能对比

| 推理引擎 | 总输出 Tokens | 执行时间 (s) | 吞吐量 (tokens/s) |
|---------|--------------|-------------|------------------|
| vLLM    | 133,966      | 98.37       | 1361.84           |
| Nano-vLLM | 133,966    | 93.41       | **1434.13**       |

**结论**：Nano-vLLM 在相同的硬件配置下，推理吞吐量超过 vLLM 约 5%。

### 性能优化建议

1. **使用 CUDA 图**：禁用 `enforce_eager=False`（默认）以启用 CUDA 图优化
2. **调整内存利用率**：根据模型大小和显存，调整 `gpu_memory_utilization` 参数
3. **前缀缓存**：相同前缀的请求会自动使用前缀缓存，提高吞吐
4. **批处理大小**：增加 `max_num_seqs` 可提高吞吐，但会增加延迟

## 代码结构

### 目录布局

```
nano-vllm/
├── nanovllm/                    # 核心包
│   ├── __init__.py             # 包入口
│   ├── llm.py                  # LLM 主类
│   ├── config.py               # 配置类
│   ├── sampling_params.py       # 采样参数
│   │
│   ├── engine/                 # 推理引擎
│   │   ├── llm_engine.py       # 引擎主类
│   │   ├── sequence.py         # 序列管理
│   │   ├── scheduler.py        # 调度器
│   │   ├── model_runner.py     # 模型执行器
│   │   └── block_manager.py    # 块管理器
│   │
│   ├── models/                 # 模型实现
│   │   └── qwen3.py           # Qwen3 模型
│   │
│   ├── layers/                 # 自定义层
│   │   ├── attention.py        # 注意力层
│   │   ├── linear.py           # 线性层（含并行）
│   │   ├── sampler.py          # 采样层
│   │   ├── layernorm.py        # 层归一化
│   │   ├── activation.py       # 激活函数
│   │   ├── embed_head.py       # 嵌入和输出头
│   │   └── rotary_embedding.py # 旋转位置编码
│   │
│   └── utils/                  # 工具函数
│       ├── loader.py           # 模型权重加载
│       └── context.py          # 执行上下文
│
├── example.py                  # 使用示例
├── bench.py                    # 基准测试脚本
├── pyproject.toml             # 项目配置
├── LICENSE                    # MIT 许可证
└── README.md                  # 本文档
```

### 核心模块详解

#### 1. engine/llm_engine.py - 推理引擎核心

```python
class LLMEngine:
    def __init__(self, model, **kwargs)
        # 初始化引擎、加载模型、创建调度器

    def add_request(self, prompt, sampling_params)
        # 添加推理请求

    def step(self)
        # 执行单步推理（前填充或解码）

    def generate(self, prompts, sampling_params)
        # 完整的生成流程
```

**关键流程**：
1. 初始化模型和调度器
2. 将请求添加到等待队列
3. 循环执行推理步骤直到所有请求完成
4. 返回解码后的文本结果

#### 2. engine/scheduler.py - 智能调度器

**调度策略**：
- **前填充阶段**：处理尽可能多的等待序列，受限于：
  - 最大序列数（`max_num_seqs`）
  - 最大批处理 token 数（`max_num_batched_tokens`）
  - 可用 KV 缓存块数

- **解码阶段**：逐 token 生成，处理已有序列
  - 若内存不足，触发序列抢占
  - 被抢占序列返回等待队列

#### 3. engine/block_manager.py - 内存管理与前缀缓存

**前缀缓存实现**：
- 使用 `xxhash` 计算每个块的哈希值
- 维护哈希到块 ID 的映射表
- 相同前缀的序列复用已缓存块
- 显著降低内存占用，提高吞吐

**示例**：
```
请求1: "问候词 + 任务描述 + 具体问题1"
请求2: "问候词 + 任务描述 + 具体问题2"
        └─── 相同前缀，块可复用 ───┘
```

#### 4. layers/ - 自定义层实现

**注意力层 (attention.py)**：
- 集成 Flash Attention 高效实现
- Triton 内核优化 KV 缓存存储

**线性层 (linear.py)**：
- 支持张量并行分片
- 列并行、行并行、复制等变体

**采样层 (sampler.py)**：
- Torch compile 优化的采样逻辑
- 基于 Gumbel-softmax 的快速采样

**位置编码 (rotary_embedding.py)**：
- RoPE（旋转位置编码）实现
- 预计算和缓存优化

#### 5. models/qwen3.py - 模型实现

实现了完整的 Qwen3 模型结构：
- 嵌入层
- 多层 Transformer 块
- 自注意力和前馈网络
- 输出头和采样

#### 6. utils/ - 工具函数

**loader.py**：
- 从 safetensors 格式加载模型权重
- 支持权重分片和张量并行映射

**context.py**：
- 线程本地执行上下文
- 存储推理时的临时信息（slot mapping、block tables 等）

## 优化技术

### 1. 前缀缓存 (Prefix Caching)

**原理**：
- 多个请求可能有相同的输入前缀
- 复用已计算的 KV 缓存，避免重复计算

**效果**：
- 减少计算量
- 降低内存占用
- 提高长上下文推理的效率

### 2. 张量并行 (Tensor Parallelism)

**实现方式**：
- 在多个 GPU 上分片模型权重
- 支持列并行、行并行等

**配置**：
```python
llm = LLM(model_path, tensor_parallel_size=4)  # 使用 4 个 GPU
```

### 3. CUDA 图优化 (CUDA Graph)

**工作原理**：
- 预先记录推理计算图
- 后续执行通过重放图，减少 CPU 开销

**配置**：
```python
llm = LLM(model_path, enforce_eager=False)  # 启用 CUDA 图（默认）
```

### 4. 动态批处理 (Dynamic Batching)

**特点**：
- 实时调度多个推理任务
- 根据资源自动调整批处理大小
- 平衡吞吐量和延迟

### 5. KV 缓存块管理

**块划分**：
- 将 KV 缓存分为固定大小的块
- 支持灵活的内存分配和回收
- 减少内存碎片

### 6. Torch Compilation

**优化目标**：
- 采样层使用 `@torch.compile` 装饰
- 自动优化计算图
- 提高执行效率

## 常见问题

### Q: Nano-vLLM 支持哪些模型？

A: Nano-vLLM 设计上支持任何标准的 Hugging Face 模型。当前已测试和优化的模型包括：
- Qwen 系列（Qwen3、Qwen2 等）
- LLaMA 系列
- 其他遵循 Transformer 标准架构的模型

### Q: 如何选择合适的配置参数？

A: 根据您的硬件资源调整：
- **GPU 内存小**（<8GB）：降低 `max_num_seqs` 和 `gpu_memory_utilization`
- **GPU 内存大**（>20GB）：增加 `max_num_batched_tokens` 和 `max_num_seqs`
- **需要低延迟**：减小批处理大小
- **需要高吞吐**：增加批处理大小

### Q: 如何在多卡 GPU 上进行推理？

A: 使用张量并行：
```python
llm = LLM(model_path, tensor_parallel_size=4)  # 4 卡并行
```

### Q: Nano-vLLM 和 vLLM 有什么区别？

A:
- **代码规模**：Nano-vLLM 约 1200 行，vLLM 数万行
- **学习难度**：Nano-vLLM 更易理解和定制
- **功能完整度**：vLLM 功能更全面（LoRA、量化等）
- **性能**：两者可比，Nano-vLLM 在某些场景更优

### Q: 如何调试和优化推理性能？

A:
1. 使用 `bench.py` 进行基准测试
2. 启用进度条了解各阶段耗时
3. 使用 NVIDIA Profiler 分析性能瓶颈
4. 根据显存占用调整内存利用率

## 开发贡献

### 报告问题

如发现 bug 或有改进建议，欢迎提交 Issue。

### 代码贡献

欢迎 Pull Request！请确保：
- 代码符合现有风格
- 包含相应的文档和测试
- 提交信息清晰明了

### 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/GeeeekExplorer/nano-vllm.git
cd nano-vllm

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装开发依赖
pip install -e ".[dev]"
```

## 参考资源

- [vLLM 官方文档](https://docs.vllm.ai/)
- [Qwen 模型说明](https://huggingface.co/Qwen)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [Triton 文档](https://triton-lang.org/)

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

### 第三方库

本项目使用了以下优秀开源库：
- PyTorch - 深度学习框架
- Transformers - 预训练模型库
- Flash Attention - 高效注意力实现
- Triton - CUDA 编程语言

## 致谢

感谢所有贡献者和社区的支持。特别感谢：
- vLLM 项目的启发和参考
- Qwen 和其他开源模型的支持
- 所有测试和反馈的用户

## 更新日志

### v0.2.0

- 改进的块管理器性能
- 增强的错误处理
- 更全面的文档

### v0.1.0

- 初始版本发布
- 支持基本推理功能
- 前缀缓存实现

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)

---

**最后更新**：2025 年 1 月

如有问题或建议，欢迎提交 Issue 或联系作者。
