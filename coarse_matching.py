"""
优化的粗筛模块 - 多度量融合
使用L2距离 + 余弦相似度 + RRF融合 + PCA降维，提高粗筛准确率
"""
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class CoarseMatcher:
    """基于全局描述符的粗筛匹配器 - 多度量融合"""

    def __init__(self, global_matrix, use_multi_metric=True, l2_weight=0.4, cosine_weight=0.6,
                 use_pca=False, pca_components=None):
        """
        初始化粗筛匹配器

        Args:
            global_matrix: 全局描述符矩阵 (N, D)
            use_multi_metric: 是否使用多度量融合
            l2_weight: L2距离权重
            cosine_weight: 余弦相似度权重
            use_pca: 是否使用PCA降维
            pca_components: PCA保留的维度（默认保留95%方差）
        """
        self.n_models = global_matrix.shape[0]
        self.dim = global_matrix.shape[1]
        self.use_multi_metric = use_multi_metric
        self.l2_weight = l2_weight
        self.cosine_weight = cosine_weight
        self.use_pca = use_pca and self.n_models > 10
        self.global_matrix_original = global_matrix.astype(np.float32)

        # PCA降维处理
        if self.use_pca:
            self._apply_pca(global_matrix, pca_components)
        else:
            self.global_matrix = self.global_matrix_original
            self.pca_matrix = None
            self.pca_mean = None

        # 特征标准化（Z-score归一化，提高度量稳定性）
        self._compute_normalization_params()

        # 归一化矩阵（用于余弦相似度）
        norms = np.linalg.norm(self.global_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 避免除零
        self.normalized_matrix = self.global_matrix / norms

        if FAISS_AVAILABLE:
            # L2距离索引
            self.index_l2 = faiss.IndexFlatL2(self.global_matrix.shape[1])
            self.index_l2.add(self.global_matrix)

            # 内积索引（用于余弦相似度，因为已归一化）
            self.index_cosine = faiss.IndexFlatIP(self.global_matrix.shape[1])
            self.index_cosine.add(self.normalized_matrix)

            self.backend = "faiss"
        else:
            self.backend = "numpy"
            print("Warning: FAISS未安装，使用NumPy后备方案")

    def _apply_pca(self, matrix, n_components=None):
        """应用PCA降维"""
        # 计算协方差矩阵
        mean = np.mean(matrix, axis=0, keepdims=True)
        centered = matrix - mean
        cov = np.dot(centered.T, centered) / (matrix.shape[0] - 1)

        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 按特征值降序排列
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 确定保留的维度
        if n_components is None:
            # 保留95%方差
            total_variance = np.sum(eigenvalues)
            cumsum = np.cumsum(eigenvalues) / total_variance
            n_components = np.searchsorted(cumsum, 0.95) + 1
            n_components = max(n_components, 10)  # 至少保留10维
            n_components = min(n_components, self.dim)

        self.pca_components = eigenvectors[:, :n_components]
        self.pca_mean = mean
        self.global_matrix = np.dot(centered, self.pca_components).astype(np.float32)

        print(f"PCA: {self.dim} -> {n_components} 维 (保留{95.0:.1f}%方差)")

    def transform(self, query):
        """将查询向量变换到PCA空间"""
        if self.use_pca and self.pca_components is not None:
            centered = query - self.pca_mean
            return np.dot(centered, self.pca_components).astype(np.float32)
        return query.astype(np.float32)

    def _compute_normalization_params(self):
        """计算特征归一化参数"""
        # 按列计算均值和标准差
        self.feature_mean = np.mean(self.global_matrix, axis=0, keepdims=True)
        self.feature_std = np.std(self.global_matrix, axis=0, keepdims=True)
        self.feature_std[self.feature_std == 0] = 1  # 避免除零

    def _normalize_features(self, features):
        """Z-score归一化特征"""
        return (features - self.feature_mean) / self.feature_std

    def search_l2(self, query_descriptor, top_k=10):
        """L2距离搜索（使用PCA变换后的特征）"""
        top_k = min(top_k, self.n_models)
        query = query_descriptor.reshape(1, -1).astype(np.float32)

        # PCA变换
        if self.use_pca:
            query = self.transform(query)

        # Z-score归一化
        query_normalized = self._normalize_features(query)

        if self.backend == "faiss":
            distances, indices = self.index_l2.search(query_normalized, top_k)
            return indices[0], distances[0]
        else:
            diffs = self.global_matrix - query_normalized
            dists = np.sum(diffs ** 2, axis=1)
            sorted_idx = np.argsort(dists)[:top_k]
            return sorted_idx, dists[sorted_idx]

    def search_cosine(self, query_descriptor, top_k=10):
        """余弦相似度搜索（使用PCA变换后的特征）"""
        top_k = min(top_k, self.n_models)

        # PCA变换
        query = query_descriptor.reshape(1, -1).astype(np.float32)
        if self.use_pca:
            query = self.transform(query)

        # 归一化查询向量
        query_norm = query / (np.linalg.norm(query) + 1e-10)

        if self.backend == "faiss":
            similarities, indices = self.index_cosine.search(query_norm.astype(np.float32), top_k)
            return indices[0], similarities[0]
        else:
            # NumPy计算余弦相似度
            query_normalized = query_norm / (np.linalg.norm(query_norm) + 1e-10)
            similarities = np.dot(self.normalized_matrix, query_normalized.T).flatten()
            sorted_idx = np.argsort(similarities)[::-1][:top_k]
            return sorted_idx, similarities[sorted_idx]

    def search_multi_metric(self, query_descriptor, top_k=10):
        """
        多度量融合搜索
        使用RRF (Reciprocal Rank Fusion) 和加权分数融合
        优化：减少重复计算，直接使用FAISS搜索
        """
        top_k = min(top_k, self.n_models)

        # 直接进行FAISS搜索，避免重复的归一化计算
        query = query_descriptor.reshape(1, -1).astype(np.float32)

        # PCA变换
        if self.use_pca:
            query = self.transform(query)

        # L2搜索（使用归一化特征）
        query_norm_l2 = self._normalize_features(query)

        # 余弦搜索（使用归一化向量）
        query_norm_cos = query / (np.linalg.norm(query) + 1e-10)

        if self.backend == "faiss":
            l2_distances, l2_indices = self.index_l2.search(query_norm_l2, self.n_models)
            cos_similarities, cos_indices = self.index_cosine.search(query_norm_cos.astype(np.float32), self.n_models)
        else:
            # NumPy后备
            diffs = self.global_matrix - query_norm_l2
            dists = np.sum(diffs ** 2, axis=1)
            l2_indices = np.argsort(dists)
            l2_distances = dists[l2_indices]

            similarities = np.dot(self.normalized_matrix, query_norm_cos.T).flatten()
            cos_indices = np.argsort(similarities)[::-1]
            cos_similarities = similarities[cos_indices]

        l2_indices = l2_indices[0]
        l2_distances = l2_distances[0]
        cos_indices = cos_indices[0]
        cos_similarities = cos_similarities[0]

        # ===== 改进的融合策略 =====

        # 方法1: RRF融合（更鲁棒）- 使用更大的k值使排名差异更明显
        rrf_k = 30  # 减小k值使排名差异更明显
        rrf_scores = np.zeros(self.n_models)

        # L2距离的RRF分数（距离越小越好，所以用倒数）
        for rank, idx in enumerate(l2_indices):
            rrf_scores[idx] += 1.0 / (rrf_k + rank + 1)

        # 余弦相似度的RRF分数（相似度越大越好）
        for rank, idx in enumerate(cos_indices):
            rrf_scores[idx] += 1.0 / (rrf_k + rank + 1)

        # 方法2: 基于分数的融合（使用实际距离/相似度值）
        # 归一化L2距离到[0,1]
        l2_scores = 1.0 - (l2_distances / (l2_distances.max() + 1e-10))

        # 余弦相似度已经在[0,1]范围
        cos_scores = cos_similarities

        # 综合两种相似度
        combined_similarities = 0.5 * l2_scores + 0.5 * cos_scores

        # 创建基于分数的排名
        score_ranked_indices = np.argsort(combined_similarities)[::-1]
        score_ranks = np.zeros(self.n_models)
        score_ranks[score_ranked_indices] = np.arange(1, self.n_models + 1)
        score_ranks_norm = score_ranks / self.n_models

        # 方法3: 加权分数融合（基于排名）
        l2_ranks = np.zeros(self.n_models)
        l2_ranks[l2_indices] = np.arange(1, self.n_models + 1)

        cos_ranks = np.zeros(self.n_models)
        cos_ranks[cos_indices] = np.arange(1, self.n_models + 1)

        l2_ranks_norm = l2_ranks / self.n_models
        cos_ranks_norm = cos_ranks / self.n_models

        fusion_scores = self.l2_weight * l2_ranks_norm + self.cosine_weight * cos_ranks_norm

        # 综合多种融合方法
        combined_scores = (
            0.4 * rrf_scores / (rrf_scores.max() + 1e-10) +
            0.3 * (1 - fusion_scores) +
            0.3 * (1 - score_ranks_norm)
        )

        # 获取top_k
        top_indices = np.argsort(combined_scores)[::-1][:top_k]

        # 返回融合后的距离（使用L2距离作为最终距离）
        final_distances = l2_distances[top_indices]

        return top_indices, final_distances

    def search(self, query_descriptor, top_k=10):
        """主搜索接口"""
        if self.use_multi_metric:
            return self.search_multi_metric(query_descriptor, top_k)
        else:
            return self.search_l2(query_descriptor, top_k)

    def __repr__(self):
        return f"CoarseMatcher(n_models={self.n_models}, dim={self.dim}, multi_metric={self.use_multi_metric})"
