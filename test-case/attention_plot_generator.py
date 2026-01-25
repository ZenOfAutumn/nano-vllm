#!/usr/bin/env python
"""
Generate visualization for Attention Q-K dot product variance analysis
生成 Attention Q-K 点积方差分析的图表
"""

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def generate_plots():
    """
    生成可视化图表
    """
    # 测试的 head_dim 维度列表：从 8 到 256，覆盖常见的模型配置
    head_dims = np.array([8, 16, 32, 64, 128, 256])

    # 用于存储无缩放时的平均方差
    variances_without_scale = []
    # 用于存储缩放后（scale=1/√d）的平均方差
    variances_with_scale = []
    # 用于存储理论方差值
    theoretical_variances = []

    # 每个维度进行的实验重复次数，越多结果越稳定
    num_experiments = 1000

    # 打印开始消息
    print("Generating plot data...")

    # 对每个 head_dim 进行分析
    for head_dim in head_dims:
        # 序列长度（模拟 Attention 中的 token 序列）
        seq_len = 32

        # 临时存储本次维度的所有无缩放方差值
        variances_no_scale = []
        # 临时存储本次维度的所有缩放后方差值
        variances_scaled = []

        # 进行多次实验
        for _ in range(num_experiments):
            # 创建查询矩阵 Q：形状 (seq_len, head_dim)
            # 每个元素从标准正态分布 N(0, 1) 采样
            Q = np.random.normal(0, 1, (seq_len, head_dim))
            # 创建键矩阵 K：形状 (seq_len, head_dim)
            # 每个元素从标准正态分布 N(0, 1) 采样
            K = np.random.normal(0, 1, (seq_len, head_dim))

            # 计算注意力分数矩阵：Q @ K^T
            # 结果形状：(seq_len, seq_len)
            # 每个元素 scores[i,j] = Σ(Q[i,k] * K[j,k])，其中 k 从 0 到 head_dim-1
            scores_no_scale = Q @ K.T

            # 计算无缩放分数的方差
            # 这个值会随 head_dim 线性增长
            var_no_scale = np.var(scores_no_scale)
            # 将该方差值添加到列表
            variances_no_scale.append(var_no_scale)

            # 计算缩放因子 scale = 1/√d
            # 这是标准 Transformer Attention 的做法
            scale = 1.0 / np.sqrt(head_dim)
            # 应用缩放因子：将所有分数乘以 1/√d
            # 这会将方差从 d 缩放到 1
            scores_scaled = scores_no_scale * scale

            # 计算缩放后分数的方差
            # 理论上应该接近 1.0
            var_scaled = np.var(scores_scaled)
            # 将该方差值添加到列表
            variances_scaled.append(var_scaled)

        # 计算无缩放方差的平均值（所有 num_experiments 次实验的平均）
        avg_var_no_scale = np.mean(variances_no_scale)
        # 计算缩放后方差的平均值
        avg_var_scaled = np.mean(variances_scaled)

        # 将该维度的平均方差添加到全局结果列表（无缩放）
        variances_without_scale.append(avg_var_no_scale)
        # 将该维度的平均方差添加到全局结果列表（缩放后）
        variances_with_scale.append(avg_var_scaled)

        # 理论方差（无缩放）：当 Q, K ~ N(0,1) 时，Q @ K^T 的方差 = head_dim
        theoretical_var_no_scale = head_dim
        # 将理论无缩放方差添加到全局结果列表
        theoretical_variances.append(theoretical_var_no_scale)

        # 打印该维度的分析进度
        print(f"  head_dim = {head_dim:3d}  ->  Var(no scale): {avg_var_no_scale:8.2f}, Var(scaled): {avg_var_scaled:6.3f}")

    # 打印开始绘图的消息
    print("\nGenerating plots...")

    # 创建 2x2 的子图布局，总图表大小为 14x10 英寸
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # 设置图表的主标题
    fig.suptitle('Attention: Q-K Dot Product Variance Analysis', fontsize=16, fontweight='bold')

    # ============ 图1：无缩放的方差随维度增长 ============
    # 获取左上角的子图对象
    ax1 = axes[0, 0]
    # 绘制实测的无缩放方差曲线（圆形标记+实线）
    ax1.plot(head_dims, variances_without_scale, 'o-', label='Measured Variance',
             linewidth=2, markersize=8, color='#1f77b4')
    # 绘制理论无缩放方差曲线（方形标记+虚线），理论值 = d
    ax1.plot(head_dims, theoretical_variances, 's--', label='Theory: Var = d',
             linewidth=2, markersize=6, color='#ff7f0e')
    # 设置 X 轴标签
    ax1.set_xlabel('Head Dimension (d)', fontsize=11, fontweight='bold')
    # 设置 Y 轴标签
    ax1.set_ylabel('Variance', fontsize=11, fontweight='bold')
    # 设置子图标题
    ax1.set_title('Variance of Q @ K^T (without scale)', fontsize=12, fontweight='bold')
    # 打开网格（透明度 0.3）
    ax1.grid(True, alpha=0.3)
    # 显示图例
    ax1.legend(fontsize=10)
    # 使用对数刻度显示 Y 轴（因为方差差异很大）
    ax1.set_yscale('log')

    # ============ 图2：缩放后的方差（应该接近 1） ============
    # 获取右上角的子图对象
    ax2 = axes[0, 1]
    # 绘制实测的缩放后方差曲线（圆形标记+实线）
    ax2.plot(head_dims, variances_with_scale, 'o-', label='Measured Variance',
             linewidth=2, markersize=8, color='#2ca02c')
    # 绘制目标方差线（水平线 y=1）
    ax2.axhline(y=1.0, color='#d62728', linestyle='--', linewidth=2,
                label='Target Variance = 1')
    # 填充 0.8 到 1.2 之间的区域作为"可接受范围"
    ax2.fill_between(head_dims, 0.8, 1.2, alpha=0.2, color='#2ca02c',
                     label='Acceptable Range')
    # 设置 X 轴标签
    ax2.set_xlabel('Head Dimension (d)', fontsize=11, fontweight='bold')
    # 设置 Y 轴标签
    ax2.set_ylabel('Variance', fontsize=11, fontweight='bold')
    # 设置子图标题
    ax2.set_title('Variance after Scaling (scale = 1/sqrt(d))', fontsize=12, fontweight='bold')
    # 打开网格
    ax2.grid(True, alpha=0.3)
    # 显示图例
    ax2.legend(fontsize=10)
    # 限制 Y 轴范围为 [0.5, 2] 方便观察
    ax2.set_ylim([0.5, 2])

    # ============ 图3：方差与维度的线性关系 ============
    # 获取左下角的子图对象
    ax3 = axes[1, 0]
    # 绘制理论线性关系（方形标记+实线），y = d
    ax3.plot(head_dims, head_dims, 's-', label='Theory: Var = d',
             linewidth=2.5, markersize=8, color='#ff7f0e')
    # 绘制实测值（圆形标记+实线）
    ax3.plot(head_dims, variances_without_scale, 'o-', label='Measured',
             linewidth=2, markersize=8, color='#1f77b4', alpha=0.7)
    # 填充理论线周围 ±10% 的区域
    ax3.fill_between(head_dims, head_dims * 0.9, head_dims * 1.1,
                     alpha=0.1, color='#ff7f0e')
    # 设置 X 轴标签
    ax3.set_xlabel('Head Dimension (d)', fontsize=11, fontweight='bold')
    # 设置 Y 轴标签
    ax3.set_ylabel('Variance', fontsize=11, fontweight='bold')
    # 设置子图标题
    ax3.set_title('Linear Growth: Variance vs Dimension', fontsize=12, fontweight='bold')
    # 打开网格（主网格和副网格）
    ax3.grid(True, alpha=0.3, which='both')
    # 显示图例
    ax3.legend(fontsize=10)

    # ============ 图4：方差变化的对数坐标 ============
    # 获取右下角的子图对象
    ax4 = axes[1, 1]
    # 使用双对数坐标绘制无缩放方差（圆形标记+实线）
    ax4.loglog(head_dims, variances_without_scale, 'o-', label='Measured (no scale)',
               linewidth=2, markersize=8, color='#1f77b4')
    # 使用双对数坐标绘制缩放后方差（方形标记+实线）
    ax4.loglog(head_dims, variances_with_scale, 's-', label='Measured (scaled)',
               linewidth=2, markersize=8, color='#2ca02c')
    # 绘制理论无缩放线 y = d（虚线）
    ax4.loglog(head_dims, head_dims, '--', label='Theory: y = d',
               linewidth=2, color='#ff7f0e', alpha=0.7)
    # 绘制理论缩放线 y = 1（虚线）
    ax4.loglog(head_dims, np.ones_like(head_dims), '--', label='Theory: y = 1 (scaled)',
               linewidth=2, color='#d62728', alpha=0.7)
    # 设置 X 轴标签
    ax4.set_xlabel('Head Dimension (d)', fontsize=11, fontweight='bold')
    # 设置 Y 轴标签
    ax4.set_ylabel('Variance', fontsize=11, fontweight='bold')
    # 设置子图标题
    ax4.set_title('Log-scale Variance Comparison', fontsize=12, fontweight='bold')
    # 打开网格（主网格和副网格都显示）
    ax4.grid(True, alpha=0.3, which='both')
    # 显示图例，最佳位置自动选择
    ax4.legend(fontsize=10, loc='best')

    # 调整子图之间的间距
    plt.tight_layout()

    # ============ 保存图表 ============
    # 获取输出路径：当前文件所在目录 + 'attention_qk_variance.png'
    output_path = os.path.join(os.path.dirname(__file__), 'attention_qk_variance.png')
    # 保存图表为 PNG 格式，DPI 150（高质量）
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    # 打印保存成功消息
    print(f"✓ Plot saved to: {output_path}")
    # 关闭图表释放内存
    plt.close()

    # 返回输出路径
    return output_path


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Generating Attention Q-K Dot Product Variance Analysis Plot")
    print("="*80 + "\n")

    output_path = generate_plots()

    print("\n" + "="*80)
    print("Done!")
    print("="*80)
    print(f"\nPlot saved to: {output_path}")
    print("\nThis plot demonstrates:")
    print("  1. Variance grows linearly with head_dim (without scaling)")
    print("  2. With scale=1/sqrt(d), variance remains constant at ~1")
    print("  3. Why Transformers use scale = 1/sqrt(d) in Attention\n")

