# BlockManager 块管理器详细指南

## 3. allocate() vs may_append() 的对比

### 核心概念

- **allocate()** - Prefill 阶段的块分配
- **may_append()** - Decode 阶段的块管理

---

## 详细对比表

| 特性 | allocate() | may_append() |
|------|-----------|------------|
| **调用阶段** | Prefill 阶段（一次） | Decode 阶段（多次） |
| **处理对象** | 整个 prompt 序列 | 单个新 token |
| **调用时机** | 推理开始时 | 每生成一个 token 后 |
| **调用次数** | 每个序列 1 次 | 每个生成的 token 1 次 |
| **块的变化** | 一次性分配所有块 | 可能添加新块 |
| **块的状态** | 块已满或不完整 | 块逐步增长 |
| **哈希表操作** | 构建完整的哈希表 | 增量更新哈希表 |
| **复杂度** | O(num_blocks) | O(1)（大多数情况） |
| **缓存复用** | 检测 prompt 块复用 | 检测生成块复用 |

---

## 详细说明

### allocate() - Prefill 阶段

#### 调用场景
```python
# 用户输入一个 prompt
prompt = "今天天气怎么样？"
seq = Sequence(tokenize(prompt))  # 256 个 token

# Prefill 阶段：一次性处理整个 prompt
if manager.can_allocate(seq):
    manager.allocate(seq)  # ✅ 分配所有块
```

#### 处理流程
```
allocate() 的执行步骤：

1. 遍历序列的所有块（可能 10 个）
   └─ 块 0 (token 0-255)
   └─ 块 1 (token 256-511)
   └─ ...
   └─ 块 9 (token 2304-2559)

2. 对每个块：
   a) 计算哈希值（用于块复用）
   b) 查找哈希表：是否已有相同块
   c) 如果有 → 复用（ref_count++）
   d) 如果无 → 分配新块

3. 一次性完成所有块分配
   result: block_table = [0, 1, 2, ..., 9]
```

#### 特点
- ✅ **批量处理**：一次处理多个块
- ✅ **可充分利用并行性**：所有块的计算可以并行化
- ✅ **块可能复用**：相同 prompt 可以共享块

#### 时间复杂度
```
O(num_blocks * num_tokens_per_block)
= O(num_tokens)

例如：2560 个 token（10 个块），O(2560) ≈ 常数时间
```

---

### may_append() - Decode 阶段

#### 调用场景
```python
# 进入 Decode 阶段：逐个生成 token
for step in range(max_steps):
    # 生成一个新 token
    output_token = model(seq)
    seq.append_token(output_token)
    seq.num_tokens += 1

    # 更新块管理
    manager.may_append(seq)  # ✅ 可能需要新块
```

#### 处理流程
```
may_append() 处理三种情况：

情况 1：len(seq) % block_size == 1（刚完成一块，开始新块）
   ├─ 最后一块已完成
   ├─ 计算其哈希值
   └─ 分配新块来存放新 token

情况 2：len(seq) % block_size == 0（刚好填满一块）
   ├─ 最后一块已满
   ├─ 计算其哈希值
   ├─ 考虑前缀哈希（与前一块的关系）
   └─ 更新哈希表，支持这个块的复用

情况 3：其他（len % block_size 在 2-255 之间）
   ├─ 块还在增长
   └─ 不需要特殊操作
```

#### 具体示例
```python
block_size = 256

# 初始状态
seq.num_tokens = 256  # 第一块已满
seq.block_table = [0]
seq.num_cached_tokens = 256

# 第 257 个 token
seq.append_token(token_257)
seq.num_tokens = 257
manager.may_append(seq)
# 257 % 256 = 1 → 情况 1
# 分配块 1
# block_table = [0, 1]

# 第 258-512 个 token
for i in range(258, 513):
    seq.append_token(token)
    seq.num_tokens += 1
    manager.may_append(seq)
    # i % 256 在 2-255 之间 → 情况 3
    # 不需要操作

# 第 512 个 token
seq.num_tokens = 512
manager.may_append(seq)
# 512 % 256 = 0 → 情况 2
# 计算块 1 的哈希，更新哈希表

# 第 513 个 token
seq.append_token(token_513)
seq.num_tokens = 513
manager.may_append(seq)
# 513 % 256 = 1 → 情况 1
# 分配块 2
# block_table = [0, 1, 2]
```

#### 特点
- ✅ **增量处理**：只处理新增的 1 个 token
- ✅ **轻量级操作**：大多数情况下不需要分配
- ✅ **动态管理**：按需分配块

#### 时间复杂度
```
平均：O(1)
最坏：O(1)（只在填满块时计算哈希）

只有当 num_tokens % block_size == 0 时才做较重操作
```

---

## LLM 推理的完整流程

### 时间轴

```
时刻 T0：收到推理请求
└─ 用户输入 prompt：500 个 token

时刻 T1：Prefill 阶段
├─ manager.allocate(seq)  ← 一次性分配所有块
├─ 分配 2 个块（500 token → 2 × 256）
├─ 建立哈希表
└─ 计算 KV 缓存：[KV_0, KV_1]

时刻 T2-T101：Decode 阶段（生成 100 个 output token）
├─ for i in range(100):
│  ├─ output_token = model(seq)
│  ├─ seq.append_token(output_token)
│  ├─ manager.may_append(seq)  ← 动态管理
│  └─ seq.num_tokens += 1
│
├─ Token 257 时（257 % 256 = 1）：分配块 3
├─ Token 512 时（512 % 256 = 0）：更新哈希表
├─ Token 513 时（513 % 256 = 1）：分配块 4
└─ ...继续生成...

时刻 T102：推理完成
└─ manager.deallocate(seq)  ← 回收所有块
```

---

## 为什么要分成两个阶段？

### Prefill 阶段的优势
- 可以批量处理大量 token
- 可以充分利用 GPU 的矩阵运算能力
- 延迟较低（几毫秒处理数百个 token）

### Decode 阶段的特点
- 每次只生成 1 个 token
- 需要多轮迭代（可能 100-1000 次）
- 对延迟敏感（需要快速响应用户）

### 性能对比
```
假设：prompt 500 token，生成 100 token

Prefill 性能：
├─ 一次处理 500 token
├─ 计算密集（矩阵乘法）
└─ 延迟：~5ms

Decode 性能：
├─ 每次处理 1 token，重复 100 次
├─ 内存访问密集（KV 缓存读取）
├─ 每次延迟：~5ms
└─ 总延迟：~500ms

总体延迟：~505ms（Prefill + Decode）
```

---

## 块复用的三个场景

### 场景 1：相同 Prompt 复用
```python
# 用户 1
prompt1 = "今天天气怎么样？"
seq1 = Sequence(tokenize(prompt1))
manager.allocate(seq1)  # 分配块 0, 1

# 用户 2（相同 prompt）
prompt2 = "今天天气怎么样？"
seq2 = Sequence(tokenize(prompt2))
manager.allocate(seq2)  # 复用块 0, 1（ref_count = 2）

# 节省显存！两个序列共享 KV 缓存
```

### 场景 2：Prompt 前缀复用
```python
# 用户 1
prompt1 = "今天天气怎么样？" + "北京"
seq1 = Sequence(tokenize(prompt1))
manager.allocate(seq1)  # 块 0, 1, 2

# 用户 2（共享前缀）
prompt2 = "今天天气怎么样？" + "上海"
seq2 = Sequence(tokenize(prompt2))
manager.allocate(seq2)  # 复用块 0, 1；分配块 3

# 共享前缀部分的 KV 缓存
```

### 场景 3：Decode 阶段的块复用
```python
# 同一序列的生成过程
# Token 256-512：建立块 1 的哈希（可能与其他序列复用）
# Token 512：计算 hash_1，更新 hash_to_block_id
# 后续：如果其他序列有相同的 token 256-512，可以复用块 1
```

---

## 关键方法的调用关系

```
推理流程：

启动
  │
  ├─ can_allocate(seq)?  ← 预检查
  │  └─ allocate(seq)    ← Prefill：一次性分配
  │
  ├─ 生成第一个 token
  │
  └─ 进入循环（Decode）
      │
      ├─ can_append(seq)?     ← 预检查（显存是否足够）
      │  └─ append_token()    ← 生成新 token
      │     └─ may_append()   ← Decode：动态管理
      │
      ├─ 检查停止条件（EOS、max_len）
      │  └─ deallocate(seq)   ← 回收块
      │
      └─ 继续循环或结束
```

---

## 总结

| 方法 | 何时调用 | 做什么 | 频率 |
|------|---------|--------|------|
| **allocate()** | Prefill 开始时 | 分配所有块 | 1 次 |
| **may_append()** | 每个 Decode token 后 | 动态管理块 | N 次 |
| **deallocate()** | 推理完成时 | 回收块 | 1 次 |

**记住**：
- 🔹 allocate() 是"一次性大工程"
- 🔹 may_append() 是"持续的小维护"
- 🔹 两者配合实现高效的 KV 缓存管理

