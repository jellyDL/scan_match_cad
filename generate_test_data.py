"""
测试数据生成工具
功能：生成模拟的CAD模型点云和扫描点云，用于验证匹配流程。
"""

import open3d as o3d
import numpy as np
import os
import argparse


def generate_random_shape(n_points=5000, shape_type="box", seed=None):
    """生成随机几何形状的点云"""
    if seed is not None:
        np.random.seed(seed)

    if shape_type == "box":
        # 随机尺寸的长方体
        dims = np.random.uniform(10, 100, size=3)
        points = np.random.uniform(-dims / 2, dims / 2, size=(n_points, 3))

    elif shape_type == "sphere":
        # 随机半径的球体
        radius = np.random.uniform(10, 50)
        phi = np.random.uniform(0, 2 * np.pi, n_points)
        theta = np.random.uniform(0, np.pi, n_points)
        r = radius * np.cbrt(np.random.uniform(0, 1, n_points))
        points = np.column_stack([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ])

    elif shape_type == "cylinder":
        # 随机尺寸的圆柱体
        radius = np.random.uniform(5, 30)
        height = np.random.uniform(20, 100)
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        r = radius * np.sqrt(np.random.uniform(0, 1, n_points))
        z = np.random.uniform(-height / 2, height / 2, n_points)
        points = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])

    else:
        # 随机点云
        points = np.random.randn(n_points, 3) * np.random.uniform(10, 50)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def generate_test_dataset(output_dir="test_data", n_models=300, n_points=5000):
    """
    生成测试数据集

    Args:
        output_dir: 输出目录
        n_models: 模型数量
        n_points: 每个模型的点数
    """
    cad_dir = os.path.join(output_dir, "cad_models")
    scan_dir = os.path.join(output_dir, "scan_data")
    os.makedirs(cad_dir, exist_ok=True)
    os.makedirs(scan_dir, exist_ok=True)

    shape_types = ["box", "sphere", "cylinder", "random"]

    print(f"生成 {n_models} 个CAD模型...")
    for i in range(n_models):
        shape = shape_types[i % len(shape_types)]
        pcd = generate_random_shape(n_points, shape, seed=i)
        path = os.path.join(cad_dir, f"model_{i:04d}.ply")
        o3d.io.write_point_cloud(path, pcd)
        if (i + 1) % 50 == 0:
            print(f"  已生成 {i+1}/{n_models}")

    # 生成扫描数据：选择一个模型，添加噪声和随机变换
    target_idx = np.random.randint(0, n_models)
    target_shape = shape_types[target_idx % len(shape_types)]
    scan_pcd = generate_random_shape(n_points, target_shape, seed=target_idx)

    # 添加高斯噪声
    points = np.asarray(scan_pcd.points)
    noise = np.random.randn(*points.shape) * 0.5
    scan_pcd.points = o3d.utility.Vector3dVector(points + noise)

    # 应用随机旋转
    R = scan_pcd.get_rotation_matrix_from_xyz(np.random.uniform(-0.1, 0.1, 3))
    scan_pcd.rotate(R, center=scan_pcd.get_center())

    scan_path = os.path.join(scan_dir, "test_scan.ply")
    o3d.io.write_point_cloud(scan_path, scan_pcd)

    print(f"\n测试数据已生成:")
    print(f"  CAD模型目录: {cad_dir} ({n_models} 个文件)")
    print(f"  扫描数据:    {scan_path}")
    print(f"  Ground Truth: model_{target_idx:04d}.ply (idx={target_idx})")

    # 保存Ground Truth
    gt_path = os.path.join(output_dir, "ground_truth.txt")
    with open(gt_path, "w") as f:
        f.write(f"target_index={target_idx}\n")
        f.write(f"target_model=model_{target_idx:04d}.ply\n")
        f.write(f"target_shape={target_shape}\n")
    print(f"  Ground Truth: {gt_path}")


def main():
    parser = argparse.ArgumentParser(description="生成测试数据集")
    parser.add_argument(
        "--output_dir", type=str, default="test_data", help="输出目录"
    )
    parser.add_argument(
        "--n_models", type=int, default=300, help="模型数量"
    )
    parser.add_argument(
        "--n_points", type=int, default=5000, help="每个模型的点数"
    )
    args = parser.parse_args()

    generate_test_dataset(args.output_dir, args.n_models, args.n_points)


if __name__ == "__main__":
    main()