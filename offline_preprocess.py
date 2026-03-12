"""
离线预处理模块
功能：对300个CAD模型进行体素降采样、法线估计、提取FPFH特征和全局描述符，
      构建特征数据库并持久化存储。
此步骤为一次性操作，不计入匹配耗时。
"""

import open3d as o3d
import numpy as np
import pickle
import os
import argparse
import time
from scipy.spatial.distance import cdist

VOXEL_SIZE = 0.5  # 体素大小，根据实际模型尺寸调整（单位与模型一致，通常mm）


def compute_d2_distribution(points, n_samples=1000, n_bins=20):
    """
    计算D2形状分布特征 - 点对距离分布

    Args:
        points: 点云坐标 (N, 3)
        n_samples: 采样点数
        n_bins: 直方图bins数量

    Returns:
        hist: 归一化的直方图分布
    """
    n_points = len(points)
    if n_points < 2:
        return np.zeros(n_bins)

    # 随机采样点对
    n_pairs = min(n_samples * (n_samples - 1) // 2, 10000)
    idx1 = np.random.choice(n_points, n_pairs, replace=True)
    idx2 = np.random.choice(n_points, n_pairs, replace=True)

    # 计算距离
    diff = points[idx1] - points[idx2]
    distances = np.sqrt(np.sum(diff ** 2, axis=1))

    # 计算直方图
    hist, _ = np.histogram(distances, bins=n_bins, range=(0, distances.max() + 1e-6))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-10)

    return hist


def compute_d1_distribution(points, n_samples=500, n_bins=15):
    """
    计算D1形状分布特征 - 点到参考点的距离分布
    对齿科修复体更敏感
    """
    n_points = len(points)
    if n_points < 2:
        return np.zeros(n_bins)

    # 使用质心作为参考点
    centroid = points.mean(axis=0)

    # 随机采样点
    sample_idx = np.random.choice(n_points, min(n_samples, n_points), replace=False)
    sample_points = points[sample_idx]

    # 计算到质心的距离
    distances = np.linalg.norm(sample_points - centroid, axis=1)

    # 计算直方图
    hist, _ = np.histogram(distances, bins=n_bins, range=(0, distances.max() + 1e-6))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-10)

    return hist


def compute_height_distribution(points, n_bins=15):
    """
    计算高度分布特征 - 对牙冠等分层结构敏感
    """
    if len(points) < 2:
        return np.zeros(n_bins)

    # 使用主成分分析确定主方向
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

    # 计算沿主方向的高度投影
    heights = np.dot(points, principal_axis)

    hist, _ = np.histogram(heights, bins=n_bins)
    hist = hist.astype(np.float32) / (hist.sum() + 1e-10)

    return hist


def compute_voxel_grid_histogram(points, grid_size=5):
    """
    计算体素网格直方图特征 - 对点云的空间分布更敏感
    """
    if len(points) < 2:
        return np.zeros(grid_size ** 3)

    # 归一化到[-1, 1]
    bbox = points.max(axis=0) - points.min(axis=0)
    center = points.mean(axis=0)
    normalized = (points - center) / (bbox.max() / 2 + 1e-10)
    normalized = np.clip(normalized, -1, 1)

    # 分配到网格
    indices = ((normalized + 1) * grid_size / 2).astype(int)
    indices = np.clip(indices, 0, grid_size - 1)

    # 计算3D直方图
    hist = np.zeros(grid_size ** 3)
    for idx in indices:
        idx_flat = idx[0] * grid_size ** 2 + idx[1] * grid_size + idx[2]
        hist[idx_flat] += 1

    hist = hist / (hist.sum() + 1e-10)
    return hist


def compute_pca_histogram(points, n_bins=10):
    """
    计算PCA特征值的直方图特征 - 捕捉形状的各向异性
    """
    if len(points) < 4:
        return np.zeros(6)

    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)

    try:
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]

        # 归一化特征值
        total = eigenvalues.sum()
        eigenvalues_norm = eigenvalues / (total + 1e-10)

        # 特征：特征值比值
        ratios = np.array([
            eigenvalues_norm[0] / (eigenvalues_norm[1] + 1e-10),
            eigenvalues_norm[1] / (eigenvalues_norm[2] + 1e-10),
            eigenvalues_norm[0] / (eigenvalues_norm[2] + 1e-10),
            eigenvalues[0] / (eigenvalues.sum() + 1e-10),
            eigenvalues[1] / (eigenvalues.sum() + 1e-10),
            eigenvalues[2] / (eigenvalues.sum() + 1e-10),
        ])

        return ratios
    except:
        return np.zeros(6)


def compute_local_point_density(points, radii=[0.5, 1.0, 2.0]):
    """
    计算局部点密度特征 - 多尺度密度
    """
    from scipy.spatial import KDTree

    if len(points) < 10:
        return np.zeros(len(radii) * 3)

    tree = KDTree(points)
    densities = []

    for radius in radii:
        counts = tree.query_ball_tree(tree, r=radius)
        counts = np.array([len(c) - 1 for c in counts])

        densities.extend([
            counts.mean(),
            counts.std(),
            counts.max()
        ])

    return np.array(densities)


def compute_local_density_variance(points, n_samples=500):
    """
    计算局部密度方差 - 对扫描数据的缺失更敏感
    """
    from scipy.spatial import KDTree

    if len(points) < 10:
        return np.zeros(4)

    # 随机采样
    if len(points) > n_samples:
        idx = np.random.choice(len(points), n_samples, replace=False)
        sample_points = points[idx]
    else:
        sample_points = points

    tree = KDTree(points)

    # 计算每个采样点的局部密度
    densities = []
    for pt in sample_points:
        neighbors = tree.query_ball_point(pt, r=1.0)
        densities.append(len(neighbors))

    densities = np.array(densities)

    return np.array([
        densities.mean(),
        densities.std(),
        densities.var(),
        np.percentile(densities, 90) / (np.percentile(densities, 10) + 1e-10)  # 密度对比度
    ])
    """
    计算局部点密度特征 - 多尺度密度

    Args:
        points: 点云坐标 (N, 3)
        radii: 多尺度半径列表

    Returns:
        densities: 各尺度的密度统计 [mean, std, max]
    """
    from scipy.spatial import KDTree

    if len(points) < 10:
        return np.zeros(len(radii) * 3)

    tree = KDTree(points)
    densities = []

    for radius in radii:
        counts = tree.query_ball_tree(tree, r=radius)
        counts = np.array([len(c) - 1 for c in counts])  # 减去中心点本身

        densities.extend([
            counts.mean(),
            counts.std(),
            counts.max()
        ])

    return np.array(densities)


def compute_surface_area_to_volume_ratio(points, triangles=None):
    """
    计算表面积与体积比近似（用于区分蜡型/牙冠等）
    使用凸包近似
    """
    from scipy.spatial import ConvexHull

    if len(points) < 4:
        return np.zeros(3)

    try:
        hull = ConvexHull(points)
        volume = hull.volume
        surface_area = hull.area

        # 避免除零
        if volume < 1e-10:
            return np.array([0.0, surface_area, 0.0])

        ratio = surface_area / (volume ** (2/3))  # 归一化
        return np.array([ratio, surface_area, volume])
    except:
        return np.zeros(3)


def compute_boundary_ratio(points):
    """
    计算边界点比例 - 牙冠边缘特征
    """
    from scipy.spatial import KDTree

    if len(points) < 10:
        return np.zeros(4)

    tree = KDTree(points)
    n_boundary = 0
    boundary_dists = []

    # 边界点：在一定距离内邻居较少的点
    for i, pt in enumerate(points):
        neighbors = tree.query_ball_point(pt, r=1.0)
        n_neighbors = len(neighbors) - 1
        if n_neighbors < 5:  # 阈值可调
            n_boundary += 1
        boundary_dists.append(n_neighbors)

    boundary_ratio = n_boundary / len(points)
    boundary_dists = np.array(boundary_dists)

    return np.array([
        boundary_ratio,
        np.mean(boundary_dists),
        np.std(boundary_dists),
        np.max(boundary_dists)
    ])


def compute_distance_to_centroid_stats(points):
    """
    计算点到质心的距离分布特征

    Args:
        points: 点云坐标 (N, 3)

    Returns:
        stats: [mean, std, max, min] 距离统计
    """
    centroid = points.mean(axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)

    return np.array([
        distances.mean(),
        distances.std(),
        distances.max(),
        distances.min()
    ])


def compute_surface_curvature_stats(points, normals, voxel_size):
    """
    计算曲率统计特征

    Args:
        points: 点云坐标
        normals: 法线
        voxel_size: 体素大小

    Returns:
        curvature_stats: 曲率统计特征
    """
    from scipy.spatial import KDTree

    if len(points) < 10:
        return np.zeros(12)

    tree = KDTree(points)
    curvatures = []

    radius = voxel_size * 3
    for i in range(len(points)):
        indices = tree.query_ball_point(points[i], r=radius)
        if len(indices) >= 3:
            # 局部协方差分析
            local_pts = points[indices]
            centered = local_pts - local_pts.mean(axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]

            # 曲率估计 = 最小特征值 / 特征值和
            curvature = eigenvalues[2] / (eigenvalues.sum() + 1e-10)
            curvatures.append(curvature)

    if len(curvatures) == 0:
        return np.zeros(12)

    curvatures = np.array(curvatures)

    # 统计特征
    return np.array([
        curvatures.mean(),
        curvatures.std(),
        curvatures.max(),
        curvatures.min(),
        np.percentile(curvatures, 25),
        np.percentile(curvatures, 50),
        np.percentile(curvatures, 75),
        np.percentile(curvatures, 90),
        np.percentile(curvatures, 95),
        np.percentile(curvatures, 99),
        (curvatures > 0.1).sum() / len(curvatures),  # 高曲率比例
        (curvatures < 0.01).sum() / len(curvatures),  # 低曲率比例
    ])


def preprocess_and_extract_features(pcd_path, voxel_size=VOXEL_SIZE):
    """
    预处理点云并提取FPFH特征 + 全局描述符

    Args:
        pcd_path: 点云文件路径（支持 .ply, .pcd, .stl, .obj 等）
        voxel_size: 体素降采样大小

    Returns:
        pcd_down: 降采样后的点云
        fpfh: FPFH局部特征
        global_descriptor: 66维全局描述符向量
    """
    ext = os.path.splitext(pcd_path)[1].lower()
    if ext == ".stl":
        mesh = o3d.io.read_triangle_mesh(pcd_path)
        print(f"Mesh info: {pcd_path} vertices={len(mesh.vertices)}, triangles={len(mesh.triangles)}")
        if mesh.is_empty() or not mesh.has_vertices():
            raise ValueError(f"无法读取STL mesh或mesh为空: {pcd_path}")
        # 使用更高的采样点数以保留更多细节
        pcd = mesh.sample_points_uniformly(number_of_points=15000)
        if pcd.is_empty():
            raise ValueError(f"STL采样后点云为空: {pcd_path}")
    else:
        pcd = o3d.io.read_point_cloud(pcd_path)
        if pcd.is_empty():
            raise ValueError(f"无法读取点云文件或点云为空: {pcd_path}")

    # 1. 体素降采样
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # 2. 法线估计
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )

    # 3. FPFH局部特征（用于后续精配准）
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
    )

    # 4. 全局描述符 = FPFH特征的统计摘要（均值 + 标准差 + 最大值 + 最小值 + 中位数 + 偏度 + 峰度）
    fpfh_data = np.asarray(fpfh.data)  # shape: (33, N)

    # 计算更多统计特征
    from scipy import stats
    mean_feat = fpfh_data.mean(axis=1)
    std_feat = fpfh_data.std(axis=1)
    max_feat = fpfh_data.max(axis=1)
    min_feat = fpfh_data.min(axis=1)
    median_feat = np.median(fpfh_data, axis=1)
    # 计算偏度和峰度（避免数值问题）
    skew_feat = stats.skew(fpfh_data, axis=1)
    kurt_feat = stats.kurtosis(fpfh_data, axis=1)
    # 处理NaN值
    skew_feat = np.nan_to_num(skew_feat, nan=0.0)
    kurt_feat = np.nan_to_num(kurt_feat, nan=0.0)

    # 5. 添加几何特征
    points = np.asarray(pcd_down.points)

    # 5.1 特征值分解
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]

    # 形状描述符
    linearity = (eigenvalues[0] - eigenvalues[1]) / (eigenvalues[0] + 1e-10)
    planarity = (eigenvalues[1] - eigenvalues[2]) / (eigenvalues[0] + 1e-10)
    sphericity = eigenvalues[2] / (eigenvalues[0] + 1e-10)

    # 5.2 点云尺度特征
    bbox = points.max(axis=0) - points.min(axis=0)
    scale = np.linalg.norm(bbox)
    aspect_ratios = bbox / (bbox.max() + 1e-10)

    geo_features = np.array([
        eigenvalues[0], eigenvalues[1], eigenvalues[2],
        linearity, planarity, sphericity,
        scale, bbox[0], bbox[1], bbox[2],
        aspect_ratios[0], aspect_ratios[1], aspect_ratios[2],
        len(points)  # 点数
    ])

    # 6. D2形状分布特征
    d2_hist = compute_d2_distribution(points, n_bins=20)

    # 7. 多尺度局部点密度特征
    density_features = compute_local_point_density(points, radii=[voxel_size * 2, voxel_size * 5, voxel_size * 10])

    # 8. 到质心距离分布特征
    centroid_dist_stats = compute_distance_to_centroid_stats(points)

    # 9. 曲率统计特征
    normals = np.asarray(pcd_down.normals)
    curvature_stats = compute_surface_curvature_stats(points, normals, voxel_size)

    # 10. 添加分位数特征（更鲁棒）
    quantiles = np.percentile(fpfh_data, [5, 10, 25, 75, 90, 95], axis=1).flatten()

    # 11. FPFH能量特征（各维度的L2范数）
    energy_feat = np.sqrt((fpfh_data ** 2).mean(axis=1))

    # 12. 添加Hausdorff距离近似特征（点到最近点的最大距离）
    from scipy.spatial import KDTree
    tree = KDTree(points)
    dists, _ = tree.query(points, k=2)
    hausdorff_approx = dists.max(axis=1)
    hausdorff_feat = np.array([
        hausdorff_approx.mean(),
        hausdorff_approx.std(),
        hausdorff_approx.max()
    ])

    # 13. D1形状分布特征（点到质心距离）
    d1_hist = compute_d1_distribution(points, n_bins=15)

    # 14. 高度分布特征
    height_hist = compute_height_distribution(points, n_bins=15)

    # 15. 表面积与体积比
    area_vol_ratio = compute_surface_area_to_volume_ratio(points)

    # 16. 边界点比例
    boundary_feat = compute_boundary_ratio(points)

    # 17. 体素网格直方图
    voxel_hist = compute_voxel_grid_histogram(points, grid_size=4)

    # 18. PCA特征值比值
    pca_ratios = compute_pca_histogram(points)

    # 19. 局部密度方差
    density_variance = compute_local_density_variance(points)

    # 组合所有特征
    global_descriptor = np.concatenate([
        mean_feat, std_feat, max_feat, min_feat, median_feat, skew_feat, kurt_feat,  # 231维
        geo_features,  # 14维
        d2_hist,  # 20维
        density_features,  # 9维
        centroid_dist_stats,  # 4维
        curvature_stats,  # 12维
        quantiles,  # 198维 (33*6)
        energy_feat,  # 33维
        hausdorff_feat,  # 3维
        d1_hist,  # 15维
        height_hist,  # 15维
        area_vol_ratio,  # 3维
        boundary_feat,  # 4维
        voxel_hist,  # 64维 (4^3)
        pca_ratios,  # 6维
        density_variance  # 4维
    ])

    return pcd_down, fpfh, global_descriptor

def build_feature_database(cad_dir, output_path="feature_db.pkl", voxel_size=VOXEL_SIZE):
    """
    构建CAD模型的特征数据库

    Args:
        cad_dir: CAD模型文件目录
        output_path: 输出的特征数据库路径
        voxel_size: 体素降采样大小

    Returns:
        db: 特征数据库字典
    """
    supported_extensions = ('.ply', '.pcd', '.stl', '.obj', '.xyz')
    db = {
        "paths": [],
        "global_descriptors": [],
        "downsampled": [],
        "fpfh": [],
        "voxel_size": voxel_size,
    }

    cad_files = sorted(
        [f for f in os.listdir(cad_dir) if f.lower().endswith(supported_extensions)]
    )

    if not cad_files:
        raise FileNotFoundError(
            f"在 {cad_dir} 中未找到支持的点云文件 ({supported_extensions})"
        )

    print(f"发现 {len(cad_files)} 个CAD模型文件，开始提取特征...")
    print("-" * 60)

    success_count = 0
    fail_count = 0

    for i, fname in enumerate(cad_files):
        fpath = os.path.join(cad_dir, fname)
        try:
            t0 = time.perf_counter()
            pcd_down, fpfh, global_desc = preprocess_and_extract_features(
                fpath, voxel_size
            )
            elapsed = time.perf_counter() - t0

            db["paths"].append(fpath)
            db["global_descriptors"].append(global_desc)
            db["downsampled"].append(np.asarray(pcd_down.points))  # 只存点云坐标数组
            db["fpfh"].append(np.asarray(fpfh.data))  # 只存numpy数组

            n_points = len(pcd_down.points)
            print(
                f"  [{i+1}/{len(cad_files)}] OK {fname} "
                f"(降采样点数: {n_points}, 耗时: {elapsed:.2f}s)"
            )
            success_count += 1

        except Exception as e:
            print(f"  [{i+1}/{len(cad_files)}] FAIL {fname} 处理失败: {e}")
            fail_count += 1

    if success_count == 0:
        raise RuntimeError("没有任何模型处理成功，请检查数据")

    # 将全局描述符堆叠为矩阵
    db["global_matrix"] = np.vstack(db["global_descriptors"]).astype(np.float32)

    # 持久化存储
    with open(output_path, "wb") as f:
        pickle.dump(db, f)

    print("-" * 60)
    print(f"特征数据库已保存: {output_path}")
    print(f"  成功: {success_count}, 失败: {fail_count}")
    print(f"  全局描述符矩阵: {db['global_matrix'].shape}")

    return db

def main():
    parser = argparse.ArgumentParser(description="离线预处理：构建CAD模型特征数据库")
    parser.add_argument(
        "--cad_dir", type=str, required=True, help="CAD模型文件目录路径"
    )
    parser.add_argument(
        "--output", type=str, default="feature_db_100_v10.pkl", help="输出特征数据库路径"
    )
    parser.add_argument(
        "--voxel_size", type=float, default=VOXEL_SIZE, help="体素降采样大小"
    )
    args = parser.parse_args()

    t_start = time.perf_counter()
    build_feature_database(args.cad_dir, args.output, args.voxel_size)
    t_total = time.perf_counter() - t_start
    print(f"总耗时: {t_total:.2f}s")

if __name__ == "__main__":
    main()