# KV 缓存分配详细指南

## 概述

`allocate_kv_cache` 函数是 LLM 推理引擎中的核心组件，负责为 GPU 分配和管理 **Key-Value (KV) 缓存**。在大模型推理中，KV 缓存的管理直接影响吞吐量、延迟和内存利用率。本文详细解释其工作原理。

---

## 1. 问题背景：为什么需要 KV 缓存？

### 1.1 自注意力机制的计算特点

在 Transformer 的自注意力机制中：
$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$

**关键观察**：
- **Query (Q)** 在每个推理步骤都会改变（因为生成了新的 token）
- **Key (K)** 和 **Value (V)** 在处理历史 token 时是**不变的**，只有新 token 的 K/V 会改变

### 1.2 缓存的必要性

在 Decode 阶段（逐 token 生成），如果不使用缓存：
- 对于长序列（如 2000 tokens），每次生成新 token 时，都需要对所有 2000 个历史 token 重新计算 K/V
- 这造成大量冗余计算和内存访问，严重降低推理速度

**使用 KV 缓存的优势**：
- 只计算新 token 的 K/V，复用历史 K/V
- 内存访问模式更友好（cache locality better）
- 推理速度可提升 **10-100 倍**

---

## 2. 内存计算详解

### 2.1 基础参数

```python
# 关键参数（来自配置和硬件）
num_kv_heads = hf_config.num_key_value_heads // self.world_size  # 张量并行下的 KV 头数
head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)  # 每个头的维度
num_hidden_layers = hf_config.num_hidden_layers  # 模型的层数
block_size = self.block_size  # 一个 KV 缓存块中的 token 数（通常为 16 或 32）
```

**数值例子**（以 Qwen3 70B 为例）：
- `hf_config.num_key_value_heads = 64`（GQA 的 KV 头数）
- `hf_config.num_hidden_layers = 80`
- `head_dim = 4096 / 64 = 64`
- `block_size = 16`（每个块包含 16 个 token）
- `world_size = 8`（张量并行的进程数）

因此：`num_kv_heads_per_gpu = 64 / 8 = 8`

### 2.2 单块 KV 缓存大小计算

**一个块的定义**：
- 一个块 = 一层的 K 或 V 缓存，包含 `block_size` 个 token
- 形状：`(block_size, num_kv_heads, head_dim)` = `(16, 8, 64)`

**字节数计算**：
$$\text{block\_bytes} = 2 \times \text{layers} \times \text{block\_size} \times \text{num\_kv\_heads} \times \text{head\_dim} \times \text{dtype\_size}$$

其中：
- **2** 因子：K 和 V 两份缓存
- **layers**：所有 80 层都需要缓存
- **dtype_size**：通常为 2（float16 或 bfloat16）

**代入数值**：
$$\text{block\_bytes} = 2 \times 80 \times 16 \times 8 \times 64 \times 2 = 2,621,440 \text{ 字节} \approx 2.5 \text{ MB}$$

### 2.3 可分配的最大块数

```python
# GPU 内存查询
free, total = torch.cuda.mem_get_info()  # free 和 total 内存（字节）
used = total - free  # 已使用内存
peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]  # 峰值分配
current = torch.cuda.memory_stats()["allocated_bytes.all.current"]  # 当前分配

# 计算可用容量
gpu_memory_utilization = config.gpu_memory_utilization  # 通常 0.9（留 10% 余地）
available_memory = int(total * gpu_memory_utilization - used - peak + current)

# 最多能分配多少个块
config.num_kvcache_blocks = available_memory // block_bytes
```

**算法逻辑**：
1. **`total * gpu_memory_utilization`**：设定的总内存利用上限（如 90GB）
2. **`- used`**：减去当前已用内存
3. **`- peak`**：减去模型权重、激活的峰值内存
4. **`+ current`**：加上当前分配（避免重复计算）
5. **`// block_bytes`**：除以单块大小，得到块数

**数值例子**（80GB GPU）：
- 总内存：80,000 MB
- 利用率：0.9 → 72,000 MB
- 模型权重（70B）：约 35,000 MB
- 激活内存：约 10,000 MB
- 可用：72,000 - 35,000 - 10,000 = 27,000 MB
- 块数：27,000 MB / 2.5 MB ≈ **10,800 块**

---

## 3. KV 缓存张量结构

### 3.1 创建缓存张量

```python
# 形状解析：(K/V, 层数, 块数, 块大小, KV头数, 头维度)
self.kv_cache = torch.empty(
    2,                              # K 和 V 两份
    hf_config.num_hidden_layers,    # 每层一份
    config.num_kvcache_blocks,      # 共 num_kvcache_blocks 个块
    self.block_size,                # 每块 block_size 个 token
    num_kv_heads,                   # num_kv_heads 个头
    head_dim                        # 每个头 head_dim 维度
)
```

**内存布局**（示例）：
```
KV 缓存张量：(2, 80, 10800, 16, 8, 64)
├─ K 缓存 [0]: 80 层，共 10800 块，每块 16 token，8 头，64 维
│  ├─ layer[0]: 10800 块，形状 (10800, 16, 8, 64)
│  ├─ layer[1]: 10800 块，形状 (10800, 16, 8, 64)
│  └─ ...
└─ V 缓存 [1]: 同上
```

**关键点**：
- 使用 `torch.empty()` 而非 `torch.zeros()`：节省初始化时间
- 张量在 **CPU 上创建**，后续按需转移到 GPU
- 总内存占用 = 块数 × 块大小 ≈ **27 GB**（对应上面的例子）

### 3.2 分配给模型层

```python
# 遍历模型的所有模块
layer_id = 0
for module in self.model.modules():
    # 找到包含 K/V 缓存的注意力层
    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        # 将缓存张量的引用赋给该层
        module.k_cache = self.kv_cache[0, layer_id]  # 形状：(num_kvcache_blocks, block_size, num_kv_heads, head_dim)
        module.v_cache = self.kv_cache[1, layer_id]  # 同上
        layer_id += 1
```

**工作原理**：
- 每个注意力层都有 `k_cache` 和 `v_cache` 属性
- 通过 **切片赋值**，让层直接访问预分配的缓存
- 这样避免重复分配，内存连续，访问高效

---

## 4. 张量并行与缓存分片

### 4.1 张量并行简介

在分布式推理中，模型被分割到多个 GPU 上：
- **总 KV 头数** = `hf_config.num_key_value_heads`（如 64）
- **单 GPU 上的 KV 头数** = `总数 / world_size`（如 64 / 8 = 8）

### 4.2 缓存分片

```python
num_kv_heads = hf_config.num_key_value_heads // self.world_size
```

**示例**（world_size = 8）：
```
GPU 0: 处理 KV 头 [0:8]
GPU 1: 处理 KV 头 [8:16]
GPU 2: 处理 KV 头 [16:24]
...
GPU 7: 处理 KV 头 [56:64]
```

**缓存分配**：
- 每个 GPU 只需分配自己负责的 KV 头部分
- 缓存大小：`(2, 80, 10800, 16, 8, 64)` → 每个 GPU 只分配 `(2, 80, 10800, 16, 8, 64)`
- **内存节省** = 原来的 1/8（注意这里 8 个 GPU 的头数不变，只是计算减少）

---

## 5. 完整流程图

```
allocate_kv_cache()
│
├─ 步骤 1: 查询 GPU 内存信息
│  ├─ 总内存 (total)
│  ├─ 已使用 (used)
│  ├─ 峰值分配 (peak)
│  └─ 当前分配 (current)
│
├─ 步骤 2: 计算单块大小
│  ├─ num_kv_heads = num_key_value_heads / world_size
│  ├─ head_dim = hidden_size / num_attention_heads
│  └─ block_bytes = 2 × layers × block_size × num_kv_heads × head_dim × dtype_size
│
├─ 步骤 3: 计算最大块数
│  └─ num_kvcache_blocks = available_memory // block_bytes
│
├─ 步骤 4: 创建 KV 缓存张量
│  └─ kv_cache.shape = (2, layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim)
│
└─ 步骤 5: 分配给模型各层
   └─ for layer_id in num_hidden_layers:
      module.k_cache = kv_cache[0, layer_id]
      module.v_cache = kv_cache[1, layer_id]
```

---

## 6. 代码详解

### 6.1 完整函数代码

```python
def allocate_kv_cache(self):
    """
    分配 KV 缓存。

    根据 GPU 内存计算能分配的最大块数，创建连续的 KV 缓存张量，
    并将其绑定到模型的各个注意力层。
    """
    config = self.config
    hf_config = config.hf_config

    # ===== 步骤 1: 查询 GPU 内存信息 =====
    free, total = torch.cuda.mem_get_info()
    used = total - free

    # 查询当前内存使用统计
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

    # ===== 步骤 2: 计算分布式情况下每个进程的参数 =====
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)

    # ===== 步骤 3: 计算单块的字节数 =====
    # 2 (K + V) × 层数 × 块大小 × 头数 × 头维度 × 数据类型字节数
    block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize

    # ===== 步骤 4: 计算能分配的最大块数 =====
    # 公式: (总内存 × 利用率 - 已用 - 峰值 + 当前) / 块大小
    config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
    assert config.num_kvcache_blocks > 0

    # ===== 步骤 5: 创建 KV 缓存张量 =====
    # 张量形状: (K/V, 层数, 块数, 块大小, 头数, 头维度)
    self.kv_cache = torch.empty(
        2,
        hf_config.num_hidden_layers,
        config.num_kvcache_blocks,
        self.block_size,
        num_kv_heads,
        head_dim
    )

    # ===== 步骤 6: 将 KV 缓存分配给模型的各个注意力层 =====
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            # 切片赋值：module 直接访问缓存的这一部分
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```

### 6.2 逐行注释

| 行号 | 代码 | 说明 |
|------|------|------|
| 1-7 | 配置和 GPU 内存查询 | 获取 GPU 内存信息，为后续计算提供基础数据 |
| 9-11 | 计算分布式参数 | 在张量并行下，每个进程只负责部分 KV 头，通过除以 `world_size` 得到 |
| 13-14 | 计算块大小 | 单块 = K 和 V 两份 × 所有层 × 块内 token 数 × 头数 × 维度 × 数据类型 |
| 16-18 | 计算块数 | 根据可用内存和块大小，计算最多能分配多少块 |
| 20-25 | 创建张量 | 预分配大块内存，避免多次小分配 |
| 27-32 | 分配给各层 | 遍历模型，将缓存张量的引用赋给各个注意力层 |

---

## 7. 实际运行示例

### 7.1 场景：Qwen3 70B on 8×80GB GPU（张量并行）

```
配置参数:
- num_key_value_heads = 64
- num_hidden_layers = 80
- hidden_size = 4096
- block_size = 16
- world_size = 8
- gpu_memory_utilization = 0.9
- dtype = torch.float16 (2 字节)

计算过程:
1. 单 GPU KV 头数: 64 / 8 = 8
2. head_dim: 4096 / 64 = 64
3. 单块大小: 2 × 80 × 16 × 8 × 64 × 2 = 2,621,440 字节 ≈ 2.5 MB

4. GPU 内存（80GB）:
   - total = 85,899,345,920 字节
   - 模型权重 ≈ 35 GB
   - 激活内存 ≈ 10 GB
   - 可用 = 85.9 × 0.9 - 35,000MB - 10,000MB ≈ 27,000 MB

5. 块数: 27,000 MB / 2.5 MB ≈ 10,800 块
6. 总序列长度: 10,800 × 16 = 172,800 tokens
```

### 7.2 KV 缓存张量维度

```python
kv_cache.shape = (2, 80, 10800, 16, 8, 64)

含义:
- 2: K 和 V 两份
- 80: 80 个 Transformer 层
- 10800: 最多缓存 10,800 个块
- 16: 每个块有 16 个 token
- 8: 每个 GPU 负责 8 个 KV 头（64 / 8）
- 64: 每个头的维度

总内存: 2 × 80 × 10800 × 16 × 8 × 64 × 2 = 282,956,185,600 字节 ≈ 27 GB
```

---

## 8. 性能影响

### 8.1 KV 缓存大小与吞吐量

| 最大序列长度 | 块数 | KV 缓存大小 | 同时处理序列数 | 预期延迟 |
|------------|------|-----------|-------------|---------|
| 4K        | 256  | 614 MB   | 50+        | 低      |
| 32K       | 2048 | 4.9 GB   | 10-20      | 中      |
| 128K      | 8192 | 19.7 GB  | 2-5        | 高      |
| 172K      | 10800| 25.9 GB  | 1-3        | 很高    |

### 8.2 优化建议

1. **平衡内存和吞吐**：不总是用满内存，留出空间给激活
2. **块大小选择**：通常 16 或 32，越大内存访问越高效，但分配粒度越粗
3. **内存利用率**：0.8-0.9 较安全，0.95+ 容易导致 OOM

---

## 9. 与其他模块的关联

### 9.1 Prefill 阶段

在 `prepare_prefill()` 中，为新 token 分配 KV 缓存槽位：

```python
# 为新 token 分配 KV 缓存槽位
for i in range(seq.num_cached_blocks, seq.num_blocks):
    start = seq.block_table[i] * self.block_size
    if i != seq.num_blocks - 1:
        end = start + self.block_size
    else:
        end = start + seq.last_block_num_tokens
    slot_mapping.extend(list(range(start, end)))
```

这些槽位索引直接对应到 `allocate_kv_cache()` 分配的缓存块。

### 9.2 Decode 阶段

在 `prepare_decode()` 中，为最后一个 token 更新 KV 缓存：

```python
# 最后一个块中最后一个 token 的槽位
slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
```

### 9.3 Block Manager

Block Manager（参见 `engine/block_manager.py`）管理块的分配和回收，决定哪些块分配给哪个序列。`allocate_kv_cache()` 提供的块池是 Block Manager 的基础资源。

---

## 10. 常见问题

### Q1: 为什么使用块而不是连续的序列缓存？

**A**: 块的好处：
- **碎片管理**：多个序列可以共享内存（一个序列用块 0-10，另一个用块 11-20）
- **重排序**：不需要移动大块数据
- **内存高效**：短序列不浪费长序列的内存

### Q2: 块大小如何选择？

**A**: 权衡因素：
- **块大小小** (8)：更灵活，但内存碎片多，访问开销大
- **块大小大** (32)：访问高效，但浪费内存，长序列碎片少

通常选择 16 作为折衷。

### Q3: 为什么 GQA（分组查询注意力）能减少 KV 缓存？

**A**: GQA 的关键：
- 多个 Query 头共享一个 Key/Value 头
- 总 KV 头数 `num_key_value_heads` 远小于 Query 头数 `num_attention_heads`
- 例：Llama 2 70B 有 64 个 Query 头但只 8 个 KV 头，缓存减少 8 倍

---

## 11. 调试与监控

### 11.1 打印调试信息

```python
def allocate_kv_cache(self):
    # ... 前面的代码 ...

    print(f"GPU Memory Info:")
    print(f"  Total: {total / 1e9:.2f} GB")
    print(f"  Used: {used / 1e9:.2f} GB")
    print(f"  Peak: {peak / 1e9:.2f} GB")
    print(f"  Current: {current / 1e9:.2f} GB")

    print(f"KV Cache Config:")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  block_bytes: {block_bytes / 1e6:.2f} MB")
    print(f"  num_kvcache_blocks: {config.num_kvcache_blocks}")
    print(f"  total kv_cache size: {config.num_kvcache_blocks * block_bytes / 1e9:.2f} GB")
    print(f"  max_seq_length: {config.num_kvcache_blocks * self.block_size}")
```

### 11.2 内存监控

```python
# 在 run() 之前和之后检查内存
before = torch.cuda.memory_allocated()
# ... 推理 ...
after = torch.cuda.memory_allocated()
print(f"推理过程中内存增长: {(after - before) / 1e9:.2f} GB")
```

---

## 12. 总结

| 方面 | 要点 |
|------|------|
| **目的** | 预分配 GPU 内存用于存储 Prefill/Decode 阶段的 K/V |
| **关键参数** | 块数、块大小、张量并行下的 KV 头数 |
| **内存计算** | 块字节数 = 2 × 层数 × 块大小 × KV头数 × 头维度 × dtype_size |
| **张量形状** | (2, layers, blocks, block_size, num_kv_heads, head_dim) |
| **分配方式** | 创建一个大张量，切片赋给各层，避免重复分配 |
| **性能影响** | 影响最大序列长度、同时处理的序列数、内存使用 |
| **常见优化** | 选择合适的块大小、利用 GQA 减少 KV 头数 |

