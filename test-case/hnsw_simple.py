"""
HNSW (Hierarchical Navigable Small World) 最简单版实现

这是一个简化版的 HNSW 算法实现，包含基本的建索引和查询功能。
HNSW 是一种高效的向量相似度搜索算法，适合大规模向量数据。

核心思想：
1. 使用多层图结构，上层节点稀疏，下层节点稠密
2. 搜索时从上层开始贪心搜索，逐层下降
3. 每层内部使用小世界网络（Small World），连接相邻和远距离节点
4. 时间复杂度：O(log N)，空间复杂度：O(N)
"""

import math
import random
from typing import List, Tuple, Set


class HNSWSimple:
    """
    HNSW 简单实现类

    属性：
        dim: 向量维度
        max_m: 每个节点最多连接的邻居数（小世界参数）
        ef_construction: 构建索引时的搜索范围（越大精度越高，构建越慢）
        ef_search: 查询时的搜索范围（越大精度越高，查询越慢）
        ml: 层数衰减因子（控制层数）
        nodes: 存储所有向量 {node_id: vector}
        graph: 多层图结构 {layer_id: {node_id: set(neighbor_ids)}}
        node_levels: 每个节点的最高层数 {node_id: max_layer}
        entry_point: 进入点（最高层的节点）
    """

    def __init__(self, dim: int, max_m: int = 5, ef_construction: int = 100, ef_search: int = 50, ml: float = 1.0 / math.log(2.0)):
        """
        初始化 HNSW 索引

        Args:
            dim: 向量维度
            max_m: 每个节点最多的邻接边数
            ef_construction: 构建索引时的搜索范围（控制建索引的精度，较大值更精确但更慢）
            ef_search: 查询时的搜索范围（控制查询的精度，较大值更精确但更慢）
            ml: 层衰减因子（通常 1/ln(2)）
        """
        # 向量维度
        self.dim = dim
        # 每个节点的最大邻接数
        self.max_m = max_m
        # 构建索引时的搜索范围（较大的值会更精确但建索引更慢）
        self.ef_construction = ef_construction
        # 查询时的搜索范围（较大的值会更精确但查询更慢）
        self.ef_search = ef_search
        # 层数衰减因子（控制图的分层结构）
        self.ml = ml

        # 存储节点的向量数据：{node_id: vector}
        self.nodes = {}
        # 多层图：{layer_id: {node_id: set(neighbors)}}
        self.graph = {}
        # 每个节点的最高层数：{node_id: layer_id}
        self.node_levels = {}
        # 进入点（用于搜索的起始点）
        self.entry_point = None
        # 当前最高层
        self.max_layer = -1

    def _distance(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量的欧几里得距离

        Args:
            vec1: 向量 1
            vec2: 向量 2

        Returns:
            欧几里得距离
        """
        # 计算向量之间的 L2 距离
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

    def _get_random_level(self) -> int:
        """
        随机生成节点的层数（遵循指数衰减分布）

        Returns:
            节点所在的最高层数
        """
        # 根据背景理论应按照指数衰减生成合理层数
        level = 0
        while random.random() < 1 / self.ml and level < 16:
            level += 1
        return level

    def _search_layer(self, query: List[float], entry_points: Set[int], num_closest: int, layer: int) -> Tuple[Set[int], Set[int]]:
        """
        在指定层进行贪心搜索（核心的小世界导航）

        Args:
            query: 查询向量
            entry_points: 起始点集合
            num_closest: 保留最近的点数
            layer: 搜索层数

        Returns:
            (已访问节点集, 最近邻点集)
        """
        # 已访问的节点集合
        visited = set()
        # 候选点集合（优先队列替代品，存储 (距离, node_id)）
        candidates = []
        # 最近邻点集合（最多保留 num_closest 个）
        w = set()
        # 最远邻点的距离（用于裁剪）
        lowerbound = 0.0

        # 初始化：将起始点加入候选集和邻点集
        for point in entry_points:
            # 计算起始点到查询向量的距离
            d = self._distance(self.nodes[point], query)
            # 将起始点加入候选集
            candidates.append((d, point))
            # 将起始点加入邻点集
            w.add(point)
            # 更新下界（最远邻点的距离）
            lowerbound = max(lowerbound, d)

        # 贪心搜索：不断扩展候选集，直到收敛
        while candidates:
            # 取出距离最小的候选点
            lowerbound_curr, nearest = min(candidates, key=lambda x: x[0])

            # 如果最近的候选点已经比当前下界更远，则搜索结束
            if lowerbound_curr > lowerbound:
                break

            # 移除已处理的候选点
            candidates.remove((lowerbound_curr, nearest))

            # 标记为已访问
            visited.add(nearest)

            # 扩展搜索：检查当前点的所有邻接点
            if layer in self.graph and nearest in self.graph[layer]:
                # 获取当前点在该层的邻接集合
                neighbors = self.graph[layer][nearest]
                for neighbor in neighbors:
                    # 跳过已访问过的节点
                    if neighbor not in visited:
                        # 计算邻接点到查询向量的距离
                        d = self._distance(self.nodes[neighbor], query)

                        # 如果距离比当前下界更近，或邻点集未满，则加入邻点集
                        if d < lowerbound or len(w) < num_closest:
                            # 加入候选点
                            candidates.append((d, neighbor))
                            # 加入邻点集
                            w.add(neighbor)
                            # 更新下界
                            lowerbound = max(lowerbound, d)

                            # 如果邻点集超过大小限制，移除最远的点
                            if len(w) > num_closest:
                                # 找到邻点集中距离最远的点
                                farthest = max(w, key=lambda x: self._distance(self.nodes[x], query))
                                # 移除最远的点
                                w.remove(farthest)
                                # 更新下界为邻点集中最远点的距离
                                lowerbound = max(self._distance(self.nodes[p], query) for p in w)

        return visited, w

    def insert(self, node_id: int, vector: List[float]):
        """
        向索引中插入一个新向量

        Args:
            node_id: 节点 ID
            vector: 向量数据
        """
        # 检查向量维度是否正确
        assert len(vector) == self.dim, f"Vector dimension {len(vector)} != {self.dim}"

        # 存储向量
        self.nodes[node_id] = vector

        # 为新节点生成随机层数
        new_level = self._get_random_level()
        # 记录节点的最高层
        self.node_levels[node_id] = new_level

        # 如果是第一个节点
        if self.entry_point is None:
            # 设置为入口点
            self.entry_point = node_id
            # 更新全局最高层
            self.max_layer = new_level
            # 为每层初始化图
            for layer in range(new_level + 1):
                if layer not in self.graph:
                    self.graph[layer] = {}
                self.graph[layer][node_id] = set()
            return

        # 从入口点开始搜索插入位置
        nearest = [self.entry_point]

        # 从最高层开始，逐层下降（跳过比新节点更高的层）
        for layer in range(min(new_level, self.max_layer), -1, -1):
            # 在当前层搜索最近的邻接点
            # ef_construction 控制搜索范围（越大搜索越彻底）
            _, nearest = self._search_layer(vector, set(nearest), 1, layer)
            nearest = list(nearest)

        # 为新节点在各层初始化
        for layer in range(new_level + 1):
            if layer not in self.graph:
                self.graph[layer] = {}
            # 在该层创建节点
            self.graph[layer][node_id] = set()

        # 如果新节点层数超过全局最高层，更新入口点和最高层
        if new_level > self.max_layer:
            # 遍历所有比新节点层数低的层
            for layer in range(self.max_layer + 1, new_level + 1):
                if layer not in self.graph:
                    self.graph[layer] = {}
                # 新建图层
                self.graph[layer][node_id] = set()
            # 更新入口点
            self.entry_point = node_id
            # 更新全局最高层
            self.max_layer = new_level

        # 在每一层中添加新节点的邻接关系
        for layer in range(min(new_level, self.max_layer) + 1):
            # 在当前层搜索 M 个最近的邻接点
            # num_closest = self.max_m 表示最多保留 max_m 个邻接
            _, candidates = self._search_layer(vector, set(nearest), self.max_m, layer)

            # 将候选点作为新节点的邻接
            self.graph[layer][node_id].update(candidates)

            # 双向建立边：候选点也要连接到新节点
            for candidate in candidates:
                # 检查候选点是否存在于该层（某些节点可能不存在于所有层）
                if candidate not in self.graph[layer]:
                    # 如果候选点不在该层，跳过
                    continue

                # 将新节点加入候选点的邻接集合
                self.graph[layer][candidate].add(node_id)

                # 如果候选点的邻接数超过 max_m，需要裁剪
                if len(self.graph[layer][candidate]) > self.max_m:
                    # 计算候选点到查询向量的距离
                    candidate_vec = self.nodes[candidate]
                    # 找到邻接集中距离最远的点（进行启发式修剪）
                    farthest = max(
                        self.graph[layer][candidate],
                        key=lambda x: self._distance(candidate_vec, self.nodes[x])
                    )
                    # 移除距离最远的邻接
                    self.graph[layer][candidate].remove(farthest)

            # 更新起始点为当前层最近的邻接点
            nearest = list(candidates)

    def search(self, query: List[float], k: int = 5) -> List[Tuple[int, float]]:
        """
        查询 k 个最近邻

        Args:
            query: 查询向量
            k: 返回最近邻的个数

        Returns:
            [(node_id, distance), ...] 排序后的最近邻列表
        """
        # 检查向量维度
        assert len(query) == self.dim, f"Query dimension {len(query)} != {self.dim}"

        # 如果索引为空，返回空列表
        if self.entry_point is None:
            return []

        # 从入口点开始搜索
        nearest = [self.entry_point]

        # 从最高层逐层下降到第 0 层
        # 上层是候选快速筛选，下层是精确搜索
        for layer in range(self.max_layer, 0, -1):
            # 在上层进行快速搜索，只保留 1 个最近点
            # 这是为了快速定位到最近的区域
            _, nearest = self._search_layer(query, set(nearest), 1, layer)
            nearest = list(nearest)

        # 在第 0 层进行精确搜索，保留 k 个最近点
        # ef_search 控制查询时的搜索范围，这里使用 max(k, self.ef_search)
        _, nearest = self._search_layer(query, set(nearest), max(k, self.ef_search), 0)

        # 计算所有候选点到查询向量的距离，并按距离排序
        results = [
            (node_id, self._distance(query, self.nodes[node_id]))
            for node_id in nearest
        ]

        # 按距离从小到大排序
        results.sort(key=lambda x: x[1])

        # 返回最近的 k 个点
        return results[:k]

