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

VOXEL_SIZE = 2.0  # 体素大小，根据实际模型尺寸调整（单位与模型一致，通常mm）

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

    # 4. 全局描述符 = FPFH特征的统计摘要（均值 + 标准差）
    fpfh_data = np.asarray(fpfh.data)  # shape: (33, N)
    global_descriptor = np.concatenate(
        [fpfh_data.mean(axis=1), fpfh_data.std(axis=1)]
    )  # 66维全局向量

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
            db["downsampled"].append(pcd_down)
            db["fpfh"].append(fpfh)

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
        "--output", type=str, default="feature_db.pkl", help="输出特征数据库路径"
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