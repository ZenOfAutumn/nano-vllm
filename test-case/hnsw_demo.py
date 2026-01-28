"""
HNSW 演示脚本

这个脚本包含 HNSW 索引的演示和性能评估函数。
"""

import random

from hnsw_simple import HNSWSimple


def calculate_average_recall(hnsw: HNSWSimple, num_queries: int = 1000, k: int = 5) -> float:
    """
    计算 HNSW 索引的平均召回准确度

    Args:
        hnsw: HNSW 索引实例
        num_queries: 执行查询的次数
        k: 每次查询返回的最近邻数量

    Returns:
        平均召回准确度（百分比）
    """
    # 累计准确度
    total_accuracy = 0.0

    # 执行多次查询
    for query_idx in range(num_queries):
        # 生成随机查询向量
        query_vector = [random.gauss(0, 1) for _ in range(hnsw.dim)]

        # 执行 HNSW 查询（近似搜索）
        hnsw_results = hnsw.search(query_vector, k=k)
        hnsw_nearest = set(node_id for node_id, _ in hnsw_results)

        # 执行暴力搜索（精确搜索，用作真实值）
        all_distances = [
            (node_id, hnsw._distance(query_vector, hnsw.nodes[node_id]))
            for node_id in hnsw.nodes
        ]
        # 排序找到真正的 k 近邻
        all_distances.sort(key=lambda x: x[1])
        true_nearest = set(node_id for node_id, _ in all_distances[:k])

        # 计算该次查询的召回准确度（Recall@k）
        # 召回率 = HNSW 找到的真正最近邻数 / 真正最近邻总数
        recall = len(true_nearest & hnsw_nearest) / k
        total_accuracy += recall

        # 每 100 次查询打印一次进度
        if (query_idx + 1) % 100 == 0:
            print(f"  已完成 {query_idx + 1}/{num_queries} 次查询...")

    # 计算平均召回准确度
    average_recall = (total_accuracy / num_queries) * 100
    return average_recall


def demo_hnsw():
    """
    演示 HNSW 的建索引和查询功能
    """
    # 设置随机种子，以便复现结果
    random.seed(42)

    # 创建 HNSW 索引：向量维度为 10
    print("=" * 60)
    print("HNSW 简单实现演示")
    print("=" * 60)

    hnsw = HNSWSimple(dim=10, max_m=5, ef_construction=100, ef_search=50)
    print(f"\n创建 HNSW 索引: dim=10, max_m=5, ef_construction=100, ef_search=50")

    # 生成随机向量并插入索引
    print("\n插入 1000 个随机向量...")
    num_vectors = 1000
    for i in range(num_vectors):
        # 生成随机向量
        vector = [random.gauss(0, 1) for _ in range(10)]
        # 插入到索引
        hnsw.insert(i, vector)
    print(f"✓ 已插入 {num_vectors} 个向量")


    # 打印每层节点数
    print("\n每层的节点数:")
    for layer in range(hnsw.max_layer + 1):
        print(f"Layer {layer}: {len(hnsw.graph[layer])} nodes")


    # 查询最近邻
    print("\n执行单次查询演示...")
    query_vector = [random.gauss(0, 1) for _ in range(10)]
    k = 5
    results = hnsw.search(query_vector, k=k)

    # 打印查询结果
    print(f"\n查询向量的最近 {k} 个邻接点:")
    print(f"{'排名':<5} {'Node ID':<12} {'距离':<12}")
    print("-" * 30)
    for rank, (node_id, distance) in enumerate(results, 1):
        print(f"{rank:<5} {node_id:<12} {distance:<12.6f}")

    # 打印索引统计信息
    print("\n索引统计信息:")
    print(f"  总节点数: {len(hnsw.nodes)}")
    print(f"  最高层: {hnsw.max_layer}")
    print(f"  入口点: {hnsw.entry_point}")
    print(f"  节点层数分布: ", end="")
    # 统计各层的节点数
    layer_counts = {}
    for node_id, level in hnsw.node_levels.items():
        layer_counts[level] = layer_counts.get(level, 0) + 1
    for layer in sorted(layer_counts.keys(), reverse=True):
        print(f"Layer {layer}: {layer_counts[layer]} nodes | ", end="")
    print()

    # 性能验证：计算单次准确度
    print("\n单次查询准确度验证:")
    # 计算所有向量到查询向量的距离（暴力搜索）
    all_distances = [
        (i, hnsw._distance(query_vector, hnsw.nodes[i]))
        for i in range(num_vectors)
    ]
    # 排序找到真正的 k 近邻
    all_distances.sort(key=lambda x: x[1])
    true_nearest = set(node_id for node_id, _ in all_distances[:k])

    # 比较 HNSW 的结果
    hnsw_nearest = set(node_id for node_id, _ in results)

    # 计算准确度（准确度 = 重合度 / k）
    accuracy = len(true_nearest & hnsw_nearest) / k * 100
    print(f"  真实最近邻: {sorted(true_nearest)}")
    print(f"  HNSW 结果:  {sorted(hnsw_nearest)}")
    print(f"  单次准确度: {accuracy:.1f}%")

    # 执行 1000 次查询统计平均召回准确度
    print("\n" + "=" * 60)
    print("执行 1000 次查询统计平均召回准确度...")
    print("=" * 60)
    average_recall = calculate_average_recall(hnsw, num_queries=1000, k=5)
    print(f"\n✓ 执行完成！")
    print(f"  平均召回准确度 (Recall@5): {average_recall:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    # 运行演示
    demo_hnsw()

