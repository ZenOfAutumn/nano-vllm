"""
HNSW ef_construction 和 ef_search 参数快速对比脚本

简化版本，用于快速演示两个参数的不同影响。
"""

import random
import time

from hnsw_simple import HNSWSimple


def quick_test(ef_construction: int, ef_search: int, num_vectors: int = 500, num_queries: int = 100) -> dict:
    """
    快速测试一个特定的参数组合

    Args:
        ef_construction: 构建时的搜索范围
        ef_search: 查询时的搜索范围
        num_vectors: 向量数量
        num_queries: 查询数量

    Returns:
        包含性能指标的字典
    """
    # 设置随机种子
    random.seed(42)

    # 创建索引
    hnsw = HNSWSimple(dim=10, max_m=5, ef_construction=ef_construction, ef_search=ef_search)

    # 构建索引
    build_start = time.time()
    for i in range(num_vectors):
        vector = [random.gauss(0, 1) for _ in range(10)]
        hnsw.insert(i, vector)
    build_time = time.time() - build_start

    # 执行查询并计算准确度
    correct_count = 0
    query_start = time.time()

    for _ in range(num_queries):
        query_vector = [random.gauss(0, 1) for _ in range(10)]

        # HNSW 查询
        hnsw_results = hnsw.search(query_vector, k=5)
        hnsw_nearest = set(node_id for node_id, _ in hnsw_results)

        # 暴力搜索（真实值）
        all_distances = [
            (node_id, hnsw._distance(query_vector, hnsw.nodes[node_id]))
            for node_id in hnsw.nodes
        ]
        all_distances.sort(key=lambda x: x[1])
        true_nearest = set(node_id for node_id, _ in all_distances[:5])

        # 统计正确数量
        correct_count += len(true_nearest & hnsw_nearest)

    query_time = time.time() - query_start
    recall = correct_count / (num_queries * 5) * 100

    return {
        "build_time": build_time,
        "query_time": query_time,
        "recall": recall,
    }


def main():
    """
    主对比测试
    """
    print("=" * 100)
    print("HNSW ef_construction 和 ef_search 参数快速对比")
    print("=" * 100)

    # 测试参数组合
    ef_construction_values = [50, 100, 200]
    ef_search_values = [25, 50, 100]

    print("\n【测试 1：ef_construction 的影响】")
    print("─" * 100)
    print("在固定 ef_search=50 的情况下，改变 ef_construction 的值")
    print()
    print(f"{'ef_construction':<20} {'构建时间(s)':<20} {'查询时间(s)':<20} {'准确度(%)':<15}")
    print("-" * 100)

    for ef_construction in ef_construction_values:
        result = quick_test(ef_construction, ef_search=50, num_vectors=500, num_queries=100)
        print(f"{ef_construction:<20} {result['build_time']:<20.3f} {result['query_time']:<20.3f} {result['recall']:<15.2f}")

    print("\n【测试 2：ef_search 的影响】")
    print("─" * 100)
    print("在固定 ef_construction=100 的情况下，改变 ef_search 的值")
    print()
    print(f"{'ef_search':<20} {'查询时间(s)':<20} {'准确度(%)':<15}")
    print("-" * 100)

    for ef_search in ef_search_values:
        result = quick_test(ef_construction=100, ef_search=ef_search, num_vectors=500, num_queries=100)
        print(f"{ef_search:<20} {result['query_time']:<20.3f} {result['recall']:<15.2f}")

    print("\n【关键发现】")
    print("─" * 100)
    print("""
1. ef_construction（构建时搜索范围）的影响：
   • 影响构建时间和索引质量
   • 值越大，构建时间越长，但索引的连接性更好
   • 这间接影响查询准确度（更好的索引 → 更高的准确度）

2. ef_search（查询时搜索范围）的影响：
   • 直接影响查询时间和准确度
   • 值越大，查询时间越长，但准确度越高
   • 这是查询时最重要的参数

3. 推荐用法：
   • 建索引时：ef_construction=100-200（取决于准确度需求）
   • 查询时：ef_search=50-100（根据实时性需求调整）
   • 可以在建索引后根据需求调整 ef_search，无需重建索引！
    """)


if __name__ == "__main__":
    main()

