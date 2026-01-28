"""
HNSW ef_construction 和 ef_search 参数对比脚本

对比不同的 ef_construction（构建索引搜索范围）和 ef_search（查询搜索范围）值对性能的影响。
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


def compare_ef_values():
    """
    对比不同 ef_construction 和 ef_search 值的性能
    """
    # 要对比的 ef_construction 值（构建时的搜索范围）
    ef_construction_values = [50, 100, 200]

    # 要对比的 ef_search 值（查询时的搜索范围）
    ef_search_values = [25, 50, 100]

    # 每个参数测试一次的参数
    num_vectors = 1000
    num_queries = 500  # 减少查询次数以节省时间
    k = 5

    # 存储结果
    results = {}

    print("=" * 100)
    print(f"HNSW ef_construction 和 ef_search 参数对比")
    print(f"（向量数={num_vectors}, 查询数={num_queries}, k={k}）")
    print("=" * 100)

    for ef_construction in ef_construction_values:
        print(f"\n{'─' * 100}")
        print(f"测试 ef_construction={ef_construction}")
        print(f"{'─' * 100}")

        # 设置随机种子确保每次都用相同的数据
        random.seed(42)

        # 创建 HNSW 索引
        hnsw = HNSWSimple(dim=10, max_m=5, ef_construction=ef_construction, ef_search=50)
        print(f"创建 HNSW 索引: dim=10, max_m=5, ef_construction={ef_construction}")

        # 记录构建开始时间
        build_start_time = time.time()

        # 插入向量
        print(f"插入 {num_vectors} 个随机向量...")
        for i in range(num_vectors):
            vector = [random.gauss(0, 1) for _ in range(10)]
            hnsw.insert(i, vector)

        # 计算构建时间
        build_time = time.time() - build_start_time
        print(f"✓ 已插入 {num_vectors} 个向量")
        print(f"  构建耗时: {build_time:.2f}s")

        # 现在测试不同的 ef_search 值
        print(f"\n对比不同 ef_search 值的查询性能:")
        print(f"{'ef_search':<12} {'准确度':<15} {'查询时间(s)':<15} {'平均每次(ms)':<15}")
        print(f"{'-' * 60}")

        for ef_search in ef_search_values:
            # 更新 ef_search
            hnsw.ef_search = ef_search

            # 执行查询并计算准确度
            query_start_time = time.time()
            average_recall = calculate_average_recall(hnsw, num_queries=num_queries, k=k)
            query_time = time.time() - query_start_time

            # 存储结果
            key = (ef_construction, ef_search)
            results[key] = {
                "recall": average_recall,
                "build_time": build_time,
                "query_time": query_time,
            }

            print(f"{ef_search:<12} {average_recall:<15.2f}% {query_time:<15.2f} {query_time / num_queries * 1000:<15.2f}")

    # 打印总结
    print("\n" + "=" * 100)
    print("总结：ef_construction 和 ef_search 对性能的影响")
    print("=" * 100)

    print("\n【ef_construction 的影响】（构建索引时的搜索范围）")
    print("─" * 100)
    print("ef_construction 值越大，构建时间越长，但索引质量越好（这会影响查询准确度）")
    print("注: 通常情况下，ef_construction 值不会直接影响查询精度，而是影响索引结构的质量")
    print()

    print("【ef_search 的影响】（查询时的搜索范围）")
    print("─" * 100)
    print(f"{'ef_construction':<20} ", end="")
    for ef_search in ef_search_values:
        print(f"ef_search={ef_search:<8} ", end="")
    print()
    print("-" * 100)

    for ef_construction in ef_construction_values:
        print(f"{ef_construction:<20} ", end="")
        for ef_search in ef_search_values:
            key = (ef_construction, ef_search)
            if key in results:
                recall = results[key]["recall"]
                print(f"{recall:>6.2f}%        ", end="")
        print()

    print("\n【关键发现】")
    print("─" * 100)

    # 分析 ef_search 的影响
    print("\n1. ef_search 对准确度的影响:")
    for ef_construction in ef_construction_values:
        recalls = []
        for ef_search in ef_search_values:
            key = (ef_construction, ef_search)
            if key in results:
                recalls.append((ef_search, results[key]["recall"]))

        if recalls:
            recalls.sort(key=lambda x: x[1])
            print(f"   ef_construction={ef_construction}:")
            print(f"     最低: ef_search={recalls[0][0]}, 准确度={recalls[0][1]:.2f}%")
            print(f"     最高: ef_search={recalls[-1][0]}, 准确度={recalls[-1][1]:.2f}%")
            print(f"     提升: {recalls[-1][1] - recalls[0][1]:.2f} 百分点")

    print("\n2. ef_search 对查询时间的影响:")
    for ef_construction in ef_construction_values:
        times = []
        for ef_search in ef_search_values:
            key = (ef_construction, ef_search)
            if key in results:
                times.append((ef_search, results[key]["query_time"]))

        if times:
            times.sort(key=lambda x: x[1])
            print(f"   ef_construction={ef_construction}:")
            print(f"     最快: ef_search={times[0][0]}, 查询时间={times[0][1]:.2f}s")
            print(f"     最慢: ef_search={times[-1][0]}, 查询时间={times[-1][1]:.2f}s")
            print(f"     增长: {(times[-1][1] / times[0][1] - 1) * 100:.1f}%")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    # 运行对比
    compare_ef_values()

