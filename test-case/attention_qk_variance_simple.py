"""
Attention 中 Q 和 K 计算结果方差分析

这个示例展示了：
1. Q @ K^T 点积结果的方差如何随维度 (head_dim) 增长而变化
2. 为什么需要缩放因子 (scale = 1/sqrt(d))
3. 缩放前后方差的对比
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def analyze_qk_variance():
    """
    分析 Q-K 点积结果的方差变化
    """
    # 维度列表：从 8 到 256，测试不同的 head_dim 值
    head_dims = np.array([8, 16, 32, 64, 128, 256])

    # 用于存储无缩放时的方差结果
    variances_without_scale = []
    # 用于存储缩放后（scale=1/√d）的方差结果
    variances_with_scale = []
    # 用于存储理论值（方差 = head_dim）
    theoretical_variances = []

    # 每个维度进行多次实验，取平均值以保证统计意义
    num_experiments = 1000

    # 打印分析开始提示
    print("\n开始分析 Q-K 点积方差随维度的变化...")
    print("=" * 80)

    # 对每个 head_dim 进行分析
    for head_dim in head_dims:
        # 假设 Q 和 K 都从标准正态分布初始化
        # 这是 Transformer 中常见的做法（即 N(0, 1)）

        # 序列长度（模拟 Attention 中的序列）
        seq_len = 32

        # 临时存储本次维度下的所有实验方差（无缩放）
        variances_no_scale = []
        # 临时存储本次维度下的所有实验方差（缩放后）
        variances_scaled = []

        # 进行多次实验以获得稳定的统计结果
        for _ in range(num_experiments):
            # Q: (seq_len, head_dim) - 查询矩阵，每个值从 N(0,1) 采样
            # K: (seq_len, head_dim) - 键矩阵，每个值从 N(0,1) 采样
            Q = np.random.normal(0, 1, (seq_len, head_dim))
            K = np.random.normal(0, 1, (seq_len, head_dim))

            # 计算 Q @ K^T 得到注意力分数（未缩放）
            # 结果形状：(seq_len, seq_len)
            # 每个元素是 head_dim 个独立随机变量乘积的和
            scores_no_scale = Q @ K.T

            # 计算无缩放情况下的方差
            # 这个值会随 head_dim 线性增长
            var_no_scale = np.var(scores_no_scale)
            variances_no_scale.append(var_no_scale)

            # 计算缩放因子 scale = 1/√d
            # 这是 Transformer 中标准做法，用于防止方差爆炸
            scale = 1.0 / np.sqrt(head_dim)
            # 应用缩放因子到分数
            # 这样可以将方差从 d 缩放到 1
            scores_scaled = scores_no_scale * scale

            # 计算缩放后的方差
            # 理论上应该接近 1.0
            var_scaled = np.var(scores_scaled)
            variances_scaled.append(var_scaled)

        # 对所有实验结果取平均，得到该维度下的平均方差（无缩放）
        avg_var_no_scale = np.mean(variances_no_scale)
        # 对所有实验结果取平均，得到该维度下的平均方差（缩放后）
        avg_var_scaled = np.mean(variances_scaled)

        # 将平均方差添加到结果列表（无缩放）
        variances_without_scale.append(avg_var_no_scale)
        # 将平均方差添加到结果列表（缩放后）
        variances_with_scale.append(avg_var_scaled)

        # 理论方差（无缩放）：Q @ K^T 中每个元素是 head_dim 个独立随机变量的和
        # 当 Q, K ~ N(0, 1) 时：
        #   E[Q_i * K_i] = 0
        #   E[(Q_i * K_i)^2] = 1
        #   Var[Σ(Q_i * K_i)] = head_dim（独立随机变量之和的方差）
        theoretical_var_no_scale = head_dim
        # 理论方差（缩放后）：由于缩放因子是 1/√d，方差会被 (1/√d)² = 1/d 倍缩放
        # 所以 Var[scaled_scores] = Var[scores / √d] = (1/d) * Var[scores] = (1/d) * d = 1
        theoretical_var_scaled = 1.0

        # 将理论无缩放方差添加到结果列表
        theoretical_variances.append(theoretical_var_no_scale)

        # 打印该维度的分析结果
        print(f"\nhead_dim = {head_dim:3d}")
        print(f"  ┌─ 无缩放方差:  {avg_var_no_scale:10.2f}  (理论值: {theoretical_var_no_scale:10.2f})")
        print(f"  └─ 缩放后方差:  {avg_var_scaled:10.2f}  (理论值: {theoretical_var_scaled:10.2f})")

    # 返回四个列表：维度、无缩放方差、缩放后方差、理论方差
    return head_dims, variances_without_scale, variances_with_scale, theoretical_variances


def plot_with_ascii():
    """
    使用 ASCII 字符绘制简单的文本图表
    """
    # 打印标题
    print("\n" + "=" * 80)
    print("ASCII 艺术图表：无缩放时方差随维度增长")
    print("=" * 80)

    # 定义要绘制的 head_dim 值
    head_dims = np.array([8, 16, 32, 64, 128, 256])
    # 注释：理论值为 方差 = head_dim（线性关系）

    # 打印坐标轴标签
    print("\n方差值 ↑")
    print("    |")

    # 用于确定图表高度的最大值
    max_var = 256

    # 从上到下遍历每个方差值级别（256, 192, ..., 0）
    for var_value in [256, 192, 128, 64, 32, 16, 8, 0]:
        # 为顶部（256）特殊处理
        if var_value == 256:
            label = "256 |"
        # 为底部（0）特殊处理
        elif var_value == 0:
            label = "  0 |"
        # 其他行采用标准格式
        else:
            label = f"{var_value:3d} |"

        # 打印当前行的纵坐标标签
        print(label, end="")
        # 对于每个 head_dim 列
        for dim in head_dims:
            # 如果当前 head_dim 的方差值 >= 当前行的值，则绘制方块（█）
            if var_value <= dim and var_value > 0:
                print(" █", end="")
            # 否则绘制点（·）表示无方差
            elif var_value == 0:
                print(" ·", end="")
            else:
                print(" ·", end="")
        # 换行
        print()

    # 打印图表底部的横轴
    # "+" 表示原点，"─" 表示坐标轴
    print("    +" + "─" * (len(head_dims) * 2 - 1))
    # 打印横轴标签（各个 head_dim 值）
    print("     " + "  ".join(str(d) for d in head_dims), "← head_dim")

    # 打印说明
    print("\n说明: 每一列代表一个 head_dim，高度代表方差")
    print("      可以看到方差随 head_dim 线性增长")


def visualize_with_matplotlib():
    """
    使用 matplotlib 绘制详细图表
    """
    # 尝试导入 matplotlib，如果不存在会捕获 ImportError 异常
    try:
        # 导入 matplotlib 库
        import matplotlib
        # 设置不需要显示图形的后端（Agg 是非交互式后端）
        matplotlib.use('Agg')
        # 导入 matplotlib 的 pyplot 模块用于绘图
        import matplotlib.pyplot as plt

        # 调用分析函数获取数据（这会再次进行数据分析）
        # 返回值：维度数组、无缩放方差、缩放后方差、理论方差
        head_dims, variances_without_scale, variances_with_scale, theoretical_variances = analyze_qk_variance()

        # 创建大图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Attention 机制中 Q-K 点积结果的方差分析', fontsize=16, fontweight='bold')

        # 图1：无缩放的方差随维度增长
        ax1 = axes[0, 0]
        ax1.plot(head_dims, variances_without_scale, 'o-', label='实测方差', linewidth=2, markersize=8, color='#1f77b4')
        ax1.plot(head_dims, theoretical_variances, 's--', label='理论方差 (= d)', linewidth=2, markersize=6, color='#ff7f0e')
        ax1.set_xlabel('Head Dimension (d)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('方差', fontsize=11, fontweight='bold')
        ax1.set_title('无缩放因子的 Q @ K^T 方差', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.set_yscale('log')

        # 图2：缩放后的方差（应该接近 1）
        ax2 = axes[0, 1]
        ax2.plot(head_dims, variances_with_scale, 'o-', label='实测方差', linewidth=2, markersize=8, color='#2ca02c')
        ax2.axhline(y=1.0, color='#d62728', linestyle='--', linewidth=2, label='目标方差 = 1')
        ax2.fill_between(head_dims, 0.8, 1.2, alpha=0.2, color='#2ca02c', label='可接受范围')
        ax2.set_xlabel('Head Dimension (d)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('方差', fontsize=11, fontweight='bold')
        ax2.set_title('缩放因子后的 Q @ K^T 方差 (scale = 1/√d)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        ax2.set_ylim([0.5, 2])

        # 图3：方差与维度的线性关系
        ax3 = axes[1, 0]
        ax3.plot(head_dims, head_dims, 's-', label='理论: 方差 = d', linewidth=2.5, markersize=8, color='#ff7f0e')
        ax3.plot(head_dims, variances_without_scale, 'o-', label='实测值', linewidth=2, markersize=8, color='#1f77b4', alpha=0.7)
        ax3.fill_between(head_dims, head_dims * 0.9, head_dims * 1.1, alpha=0.1, color='#ff7f0e')
        ax3.set_xlabel('Head Dimension (d)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('方差', fontsize=11, fontweight='bold')
        ax3.set_title('无缩放时方差与维度的线性关系', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, which='both')
        ax3.legend(fontsize=10)

        # 图4：方差变化的对数坐标
        ax4 = axes[1, 1]
        ax4.loglog(head_dims, variances_without_scale, 'o-', label='实测方差 (无缩放)', linewidth=2, markersize=8, color='#1f77b4')
        ax4.loglog(head_dims, variances_with_scale, 's-', label='实测方差 (缩放)', linewidth=2, markersize=8, color='#2ca02c')
        ax4.loglog(head_dims, head_dims, '--', label='理论: y = d', linewidth=2, color='#ff7f0e', alpha=0.7)
        ax4.loglog(head_dims, np.ones_like(head_dims), '--', label='理论: y = 1 (缩放后)', linewidth=2, color='#d62728', alpha=0.7)
        ax4.set_xlabel('Head Dimension (d)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('方差', fontsize=11, fontweight='bold')
        ax4.set_title('对数坐标下的方差变化', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, which='both')
        ax4.legend(fontsize=10, loc='best')

        # 调整子图之间的间距以防止重叠
        plt.tight_layout()

        # ============ 保存图表到文件 ============
        # 构建输出路径：当前脚本目录 + 文件名
        output_path = os.path.join(os.path.dirname(__file__), 'attention_qk_variance.png')
        # 保存图表为 PNG 格式，DPI=150 表示高清晰度
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        # 打印保存成功的信息
        print(f"\n✓ 图表已保存到: {output_path}")
        # 关闭图表以释放内存
        plt.close()

    # 捕获 ImportError 异常（matplotlib 未安装）
    except ImportError:
        # 打印警告消息
        print("\n⚠ matplotlib 未安装，跳过 matplotlib 可视化")
        # 提供安装建议
        print("  如需生成图表，请运行: pip install matplotlib")


def explain_key_concepts():
    """
    打印关键概念说明
    """
    print("\n" + "="*80)
    print("核心概念说明")
    print("="*80)

    explanation = """
1️⃣  Q-K 点积的方差增长问题：
    ├─ Q 和 K 都是从 N(0,1) 初始化
    ├─ Q @ K^T 中每个元素是 head_dim 个独立乘积的和
    └─ 结果方差 = head_dim (线性增长)

2️⃣  为什么需要缩放因子 (1/√d)？
    ├─ 无缩放时，维度越大，方差越大
    ├─ Softmax 对大值不敏感，会丧失梯度信息
    ├─ 使用 scale = 1/√d 可以使方差保持在 ~1
    └─ 这样 Softmax 就能充分利用动态范围

3️⃣  缩放后的效果：
    ├─ 无论 head_dim 多大，缩放后方差都接近 1
    ├─ 保持数值稳定性
    ├─ 使梯度流动更平滑
    └─ 这就是 Transformer 中 scale = 1/√d 的由来

4️⃣  实际应用：
    ├─ Attention scores = softmax(Q @ K^T / √d)
    ├─ 大多数 Transformer 实现都使用这个缩放因子
    ├─ 对于 head_dim 较大时尤其重要
    └─ 在 nano-vllm 中由 Attention 层的 scale 参数控制
    """
    print(explanation)


def torch_implementation_example():
    """
    展示实现相关的代码
    """
    print("\n" + "="*80)
    print("实现示例与对应文件")
    print("="*80)

    code_example = """
📄 Attention 的典型实现流程：

1. 初始化阶段（nanovllm/layers/attention.py）:
   ├─ __init__: 设置 scale = 1/√d
   └─ 这个 scale 是关键参数

2. Forward 前向传播:
   ├─ 接收 Q, K, V 输入
   ├─ 存储 KV 缓存
   └─ 调用 flash_attn，传入 softmax_scale=self.scale

3. 核心计算流程:
   ├─ scores = Q @ K^T         (方差会爆炸 ∝ head_dim)
   ├─ scores = scores * scale   (除以 √d，方差恢复到 1)
   ├─ attn_weights = softmax(scores)
   └─ output = attn_weights @ V

4. nano-vllm 中的具体代码:
   文件: nanovllm/layers/attention.py

   class Attention(nn.Module):
       def __init__(self, num_heads, head_dim, scale, num_kv_heads):
           self.scale = scale  # 这里就是 1/√d

       def forward(self, q, k, v):
           # flash_attn 会使用 softmax_scale 参数
           o = flash_attn_varlen_func(
               q, k, v,
               softmax_scale=self.scale,  # ← 关键！
               ...
           )
    """
    print(code_example)


def mathematical_derivation():
    """
    数学推导
    """
    print("\n" + "="*80)
    print("数学推导")
    print("="*80)

    derivation = """
假设 Q, K 都从 N(0, 1) 初始化：

💡 第一步：计算点积的期望和方差
   对于单个元素 scores[i,j] = Q[i] · K[j]^T

   scores[i,j] = Σ(Q[i,k] * K[j,k])  其中 k ∈ [1, d]

   由于每个 Q[i,k] 和 K[j,k] 都是 N(0, 1)：
   - E[Q[i,k] * K[j,k]] = 0
   - Var[Q[i,k] * K[j,k]] = 1

   因此：
   E[scores[i,j]] = 0
   Var[scores[i,j]] = d  (d 个独立随机变量之和)

💡 第二步：方差随维度线性增长
   dim=8   → Var ≈ 8
   dim=64  → Var ≈ 64
   dim=256 → Var ≈ 256

   这导致数值变得极端，Softmax 失效！

💡 第三步：应用缩放因子
   scaled_scores = scores / √d

   Var[scaled_scores] = Var[scores / √d]
                       = (1/d) * Var[scores]
                       = (1/d) * d
                       = 1

   完美！方差现在恒定为 1

💡 第四步：Softmax 的最佳工作范围
   Softmax 在输入值在 [-3, 3] 范围内时表现最好
   - 较小的值：梯度可以充分流动
   - 较大的值：Softmax 退化为 one-hot（梯度消失）

   使用 scale=1/√d 确保输入值保持在合理范围内
    """
    print(derivation)


def summary():
    """
    总结
    """
    print("\n" + "="*80)
    print("总结：为什么 Transformer 需要 scale = 1/√d")
    print("="*80)

    summary_text = """
┌─────────────────────────────────────────────────────────────────────┐
│ 问题：Q-K 点积会产生方差爆炸                                          │
│ ─────────────────────────────────────────────────────────────────── │
│ • head_dim 越大，方差越大（线性关系）                                │
│ • 导致 Softmax 退化，梯度消失                                        │
│ • 模型训练不稳定，收敛速度慢                                         │
├─────────────────────────────────────────────────────────────────────┤
│ 解决方案：使用缩放因子 scale = 1/√d                                  │
│ ─────────────────────────────────────────────────────────────────── │
│ • 将方差从 d 缩放到 1                                               │
│ • 保持数值稳定性                                                    │
│ • Softmax 在最优范围内工作                                          │
│ • 梯度流动平滑，训练收敛快                                          │
├─────────────────────────────────────────────────────────────────────┤
│ 结果：现代 Transformer 的标准做法                                     │
│ ─────────────────────────────────────────────────────────────────── │
│ ✓ 所有主流模型都使用这个技巧                                         │
│ ✓ 包括 GPT, BERT, Qwen 等                                           │
│ ✓ nano-vllm 也使用 flash_attn 的 softmax_scale 参数实现              │
└─────────────────────────────────────────────────────────────────────┘
    """
    print(summary_text)


if __name__ == "__main__":
    # 打印脚本标题和分隔线
    print("\n" + "="*80)
    print("Attention 机制中 Q-K 点积方差分析")
    print("="*80)

    # 第一步：运行核心分析（计算各种 head_dim 下的方差）
    analyze_qk_variance()

    # 第二步：绘制 ASCII 文本图表（用 ASCII 字符显示方差趋势）
    plot_with_ascii()

    # 第三步：尝试生成 matplotlib 详细图表（如果安装了 matplotlib）
    visualize_with_matplotlib()

    # 第四步：解释核心概念
    explain_key_concepts()

    # 第五步：展示数学推导
    mathematical_derivation()

    # 第六步：展示 PyTorch/Transformer 实现示例
    torch_implementation_example()

    # 第七步：总结为什么使用这个缩放因子
    summary()

    # 打印完成消息
    print("\n✨ 分析完成！\n")

