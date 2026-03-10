"""
粗筛模块
功能：使用FAISS对全局描述符进行快速近似最近邻搜索，
      从300个CAD模型中筛选出Top-K候选。
预估耗时：< 1ms
"""

import numpy as np

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class CoarseMatcher:
    """
    基于全局描述符的粗筛匹配器

    使用FAISS进行高效的L2距离最近邻搜索。
    对于300个模型，直接使用暴力搜索（IndexFlatL2）即可在<1ms内完成。
    如果模型数量 >10000，建议切换到 IndexIVFFlat 或 IndexHNSW。
    """

    def __init__(self, global_matrix):
        """
        初始化粗筛匹配器

        Args:
            global_matrix: shape (N, D) 的特征矩阵，
                          N=模型数量, D=描述符维度(66)
        """
        self.n_models = global_matrix.shape[0]
        self.dim = global_matrix.shape[1]
        self.global_matrix = global_matrix.astype(np.float32)

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(self.dim)
            self.index.add(self.global_matrix)
            self.backend = "faiss"
        else:
            # 回退到NumPy暴力搜索
            self.backend = "numpy"
            print("Warning: FAISS未安装，使用NumPy后备方案（速度略慢但功能相同）")

    def search(self, query_descriptor, top_k=10):
        """
        搜索最近邻

        Args:
            query_descriptor: shape (D,) 的查询描述符
            top_k: 返回的候选数量

        Returns:
            indices: Top-K候选的索引数组
            distances: 对应的L2距离
        """
        top_k = min(top_k, self.n_models)
        query = query_descriptor.reshape(1, -1).astype(np.float32)

        if self.backend == "faiss":
            distances, indices = self.index.search(query, top_k)
            return indices[0], distances[0]
        else:
            # NumPy暴力搜索
            diffs = self.global_matrix - query
            dists = np.sum(diffs ** 2, axis=1)
            sorted_idx = np.argsort(dists)[:top_k]
            return sorted_idx, dists[sorted_idx]

    def __repr__(self):
        return (
            f"CoarseMatcher(n_models={{self.n_models}}, "
            f"dim={{self.dim}}, backend={{self.backend}})"
        )
