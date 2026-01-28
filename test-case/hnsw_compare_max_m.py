"""
HNSW max_m 参数对比脚本

对比不同 max_m 值对召回准确度的影响。
max_m 是每个节点最多连接的邻接数，更大的值会增加索引大小但可能提升准确度。
"""

import random
import time

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
        recall = len(true_nearest & hnsw_nearest) / k
        total_accuracy += recall

    # 计算平均召回准确度
    average_recall = (total_accuracy / num_queries) * 100
    return average_recall


def compare_max_m_values():
    """
    对比不同 max_m 值的性能
    """
    # 要对比的 max_m 值
    max_m_values = [5, 6, 7]
    # 每个参数测试一次的参数
    num_vectors = 1000
    num_queries = 1000
    k = 5

    # 存储结果
    results = {}

    print("=" * 80)
    print(f"HNSW max_m 参数对比（向量数={num_vectors}, 查询数={num_queries}, k={k}）")
    print("=" * 80)

    for max_m in max_m_values:
        print(f"\n{'─' * 80}")
        print(f"测试 max_m={max_m}")
        print(f"{'─' * 80}")

        # 设置随机种子确保每次都用相同的数据
        random.seed(42)

        # 创建 HNSW 索引
        hnsw = HNSWSimple(dim=10, max_m=max_m, ef_construction=100, ef_search=50)
        print(f"创建 HNSW 索引: dim=10, max_m={max_m}, ef_construction=100, ef_search=50")

        # 记录构建开始时间
        build_start_time = time.time()

        # 插入向量
        print(f"\n插入 {num_vectors} 个随机向量...")
        for i in range(num_vectors):
            vector = [random.gauss(0, 1) for _ in range(10)]
            hnsw.insert(i, vector)

        # 计算构建时间
        build_time = time.time() - build_start_time
        print(f"✓ 已插入 {num_vectors} 个向量")
        print(f"  构建耗时: {build_time:.2f}s")

        # 打印索引统计信息
        print(f"\n索引统计信息:")
        print(f"  总节点数: {len(hnsw.nodes)}")
        print(f"  最高层: {hnsw.max_layer}")

        # 计算总边数（用于估计索引大小）
        total_edges = 0
        for layer in hnsw.graph.values():
            for neighbors in layer.values():
                total_edges += len(neighbors)
        print(f"  总边数: {total_edges}")
        print(f"  平均每个节点的边数: {total_edges / len(hnsw.nodes):.2f}")

        # 执行查询并计算准确度
        print(f"\n执行 {num_queries} 次查询...")
        query_start_time = time.time()
        average_recall = calculate_average_recall(hnsw, num_queries=num_queries, k=k)
        query_time = time.time() - query_start_time

        print(f"✓ 查询完成")
        print(f"  查询耗时: {query_time:.2f}s")
        print(f"  平均每次查询: {query_time / num_queries * 1000:.2f}ms")

        # 存储结果
        results[max_m] = {
            "recall": average_recall,
            "build_time": build_time,
            "query_time": query_time,
            "total_edges": total_edges,
        }

    # 打印对比总结
    print("\n" + "=" * 80)
    print("对比总结")
    print("=" * 80)
    print(f"\n{'max_m':<8} {'召回准确度':<15} {'构建时间(s)':<15} {'查询时间(s)':<15} {'总边数':<10}")
    print("-" * 80)

    for max_m in max_m_values:
        result = results[max_m]
        print(f"{max_m:<8} {result['recall']:<15.2f}% {result['build_time']:<15.2f} {result['query_time']:<15.2f} {result['total_edges']:<10}")

    # 分析结果
    print("\n" + "=" * 80)
    print("性能分析")
    print("=" * 80)

    # 找到最高准确度
    best_recall_max_m = max(results.keys(), key=lambda x: results[x]["recall"])
    worst_recall_max_m = min(results.keys(), key=lambda x: results[x]["recall"])

    best_recall = results[best_recall_max_m]["recall"]
    worst_recall = results[worst_recall_max_m]["recall"]
    recall_improvement = best_recall - worst_recall

    print(f"\n✓ 最高准确度: max_m={best_recall_max_m}, 准确度={best_recall:.2f}%")
    print(f"✗ 最低准确度: max_m={worst_recall_max_m}, 准确度={worst_recall:.2f}%")
    print(f"△ 准确度提升: {recall_improvement:.2f} 百分点")

    # 分析构建时间
    print(f"\n构建时间对比:")
    fastest_build_max_m = min(results.keys(), key=lambda x: results[x]["build_time"])
    slowest_build_max_m = max(results.keys(), key=lambda x: results[x]["build_time"])

    fastest_build_time = results[fastest_build_max_m]["build_time"]
    slowest_build_time = results[slowest_build_max_m]["build_time"]
    build_overhead = (slowest_build_time - fastest_build_time) / fastest_build_time * 100

    print(f"  最快构建: max_m={fastest_build_max_m}, 耗时={fastest_build_time:.2f}s")
    print(f"  最慢构建: max_m={slowest_build_max_m}, 耗时={slowest_build_time:.2f}s")
    print(f"  时间开销: {build_overhead:.2f}%")

    # 分析查询时间
    print(f"\n查询时间对比:")
    fastest_query_max_m = min(results.keys(), key=lambda x: results[x]["query_time"])
    slowest_query_max_m = max(results.keys(), key=lambda x: results[x]["query_time"])

    fastest_query_time = results[fastest_query_max_m]["query_time"]
    slowest_query_time = results[slowest_query_max_m]["query_time"]
    query_overhead = (slowest_query_time - fastest_query_time) / fastest_query_time * 100

    print(f"  最快查询: max_m={fastest_query_max_m}, 耗时={fastest_query_time:.2f}s")
    print(f"  最慢查询: max_m={slowest_query_max_m}, 耗时={slowest_query_time:.2f}s")
    print(f"  时间开销: {query_overhead:.2f}%")

    # 分析边数（索引大小）
    print(f"\n索引大小对比（以边数计）:")
    min_edges_max_m = min(results.keys(), key=lambda x: results[x]["total_edges"])
    max_edges_max_m = max(results.keys(), key=lambda x: results[x]["total_edges"])

    min_edges = results[min_edges_max_m]["total_edges"]
    max_edges = results[max_edges_max_m]["total_edges"]
    edge_overhead = (max_edges - min_edges) / min_edges * 100

    print(f"  最小索引: max_m={min_edges_max_m}, 边数={min_edges}")
    print(f"  最大索引: max_m={max_edges_max_m}, 边数={max_edges}")
    print(f"  大小开销: {edge_overhead:.2f}%")

    # 建议
    print("\n" + "=" * 80)
    print("建议")
    print("=" * 80)

    avg_recall = sum(results[m]["recall"] for m in max_m_values) / len(max_m_values)

    for max_m in max_m_values:
        recall = results[max_m]["recall"]
        build_time = results[max_m]["build_time"]
        query_time = results[max_m]["query_time"]

        print(f"\nmax_m={max_m}:")

        if recall > avg_recall:
            print(f"  ✓ 准确度较高 ({recall:.2f}%)")
        elif recall < avg_recall:
            print(f"  ✗ 准确度较低 ({recall:.2f}%)")
        else:
            print(f"  = 准确度平均 ({recall:.2f}%)")

        if build_time < results[fastest_build_max_m]["build_time"] * 1.05:
            print(f"  ✓ 构建速度较快 ({build_time:.2f}s)")
        else:
            print(f"  ✗ 构建速度较慢 ({build_time:.2f}s)")

        if query_time < results[fastest_query_max_m]["query_time"] * 1.05:
            print(f"  ✓ 查询速度较快 ({query_time:.2f}s)")
        else:
            print(f"  ✗ 查询速度较慢 ({query_time:.2f}s)")


if __name__ == "__main__":
    # 运行对比
    compare_max_m_values()

