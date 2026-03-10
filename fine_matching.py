"""
精匹配模块
功能：对粗筛阶段返回的Top-K候选，逐一进行
      Fast Global Registration (FGR) + ICP 精细配准，
      根据 fitness 和 RMSE 选出最佳匹配。
预估耗时：每个候选 100~200ms * 10候选 = 1~2秒
"""

import open3d as o3d
import numpy as np


def fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """
    快速全局配准 (Fast Global Registration)

    Args:
        source_down: 源点云（降采样后）
        target_down: 目标点云（降采样后）
        source_fpfh: 源点云的FPFH特征
        target_fpfh: 目标点云的FPFH特征
        voxel_size: 体素大小

    Returns:
        配准结果 (RegistrationResult)
    """
    distance_threshold = voxel_size * 0.5
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        ),
    )
    return result


def refine_with_icp(source_down, target_down, init_transform, voxel_size):
    """
    ICP精细配准

    Args:
        source_down: 源点云
        target_down: 目标点云
        init_transform: 初始变换矩阵（来自FGR）
        voxel_size: 体素大小

    Returns:
        配准结果 (RegistrationResult)
    """
    distance_threshold = voxel_size * 0.2
    result = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        distance_threshold,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
    )
    return result


def fine_match(scan_down, scan_fpfh, candidates, db, voxel_size):
    """
    对粗筛候选进行精匹配，选出最佳匹配

    Args:
        scan_down: 扫描点云（降采样后）
        scan_fpfh: 扫描点云的FPFH特征
        candidates: 粗筛返回的候选索引数组
        db: 特征数据库
        voxel_size: 体素大小

    Returns:
        dict: 包含最佳匹配信息的字典
            - best_index: 最佳匹配在数据库中的索引
            - best_path: 最佳匹配的文件路径
            - fitness: 配准质量（内点比例，0~1）
            - rmse: 内点的均方根误差
            - transformation: 4x4变换矩阵
            - all_results: 所有候选的评估结果列表
    """
    best_idx = -1
    best_fitness = -1.0
    best_rmse = float("inf")
    best_transform = None
    all_results = []

    for rank, idx in enumerate(candidates):
        cad_down = db["downsampled"][idx]
        cad_fpfh = db["fpfh"][idx]

        try:
            # Step 1: Fast Global Registration
            fgr_result = fast_global_registration(
                scan_down, cad_down, scan_fpfh, cad_fpfh, voxel_size
            )

            # Step 2: ICP 精配准
            icp_result = refine_with_icp(
                scan_down, cad_down, fgr_result.transformation, voxel_size
            )

            result_info = {
                "rank": rank,
                "index": idx,
                "path": db["paths"][idx],
                "fitness": icp_result.fitness,
                "rmse": icp_result.inlier_rmse,
                "transformation": icp_result.transformation,
            }
            all_results.append(result_info)

            # 评估：fitness越高越好，相同fitness下RMSE越低越好
            if icp_result.fitness > best_fitness or (
                icp_result.fitness == best_fitness
                and icp_result.inlier_rmse < best_rmse
            ):
                best_fitness = icp_result.fitness
                best_rmse = icp_result.inlier_rmse
                best_idx = idx
                best_transform = icp_result.transformation

        except Exception as e:
            print(f"  Warning: 候选 #{rank} (idx={idx}) 配准失败: {e}")
            all_results.append(
                {
                    "rank": rank,
                    "index": idx,
                    "path": db["paths"][idx],
                    "fitness": 0.0,
                    "rmse": float("inf"),
                    "error": str(e),
                }
            )

    if best_idx == -1:
        raise RuntimeError("所有候选配准均失败，无法找到匹配")

    return {
        "best_index": best_idx,
        "best_path": db["paths"][best_idx],
        "fitness": best_fitness,
        "rmse": best_rmse,
        "transformation": best_transform,
        "all_results": sorted(all_results, key=lambda x: (-x["fitness"], x["rmse"])),
    }
