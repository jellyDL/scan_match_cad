"""
优化的精匹配模块 - 平衡准确率和速度
"""
import open3d as o3d
import numpy as np


def fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """快速全局配准"""
    distance_threshold = voxel_size * 0.5
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        ),
    )
    return result


def refine_with_icp(source_down, target_down, init_transform, voxel_size, use_p2p=False):
    """ICP精细配准"""
    distance_threshold = voxel_size * 0.3
    if use_p2p:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, distance_threshold, init_transform,
        estimation,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
    )
    return result


def fine_match_all(scan_down, scan_fpfh, db, voxel_size, coarse_distances=None):
    """对所有CAD模型进行精匹配 - 带回退机制"""
    best_idx = -1
    best_fitness = -1.0
    best_rmse = float("inf")
    best_transform = None
    all_results = []

    n_models = len(db["paths"])
    print(f"  对所有 {n_models} 个模型进行配准...")

    # 归一化粗筛距离
    if coarse_distances is not None:
        max_dist = max(coarse_distances) + 1e-10
        norm_dist = [d / max_dist for d in coarse_distances]
    else:
        norm_dist = [0] * n_models

    for rank, idx in enumerate(range(n_models)):
        cad_down_np = db["downsampled"][idx]
        cad_down = o3d.geometry.PointCloud()
        cad_down.points = o3d.utility.Vector3dVector(cad_down_np)
        cad_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )

        cad_fpfh_np = db["fpfh"][idx]
        cad_fpfh = o3d.pipelines.registration.Feature()
        cad_fpfh.data = cad_fpfh_np.astype(np.float64)

        try:
            fgr_result = fast_global_registration(scan_down, cad_down, scan_fpfh, cad_fpfh, voxel_size)
            icp_result = refine_with_icp(scan_down, cad_down, fgr_result.transformation, voxel_size)

            result_info = {
                "rank": rank,
                "index": idx,
                "path": db["paths"][idx],
                "fitness": icp_result.fitness,
                "rmse": icp_result.inlier_rmse,
                "correspondence_set_size": len(icp_result.correspondence_set),
                "transformation": icp_result.transformation,
            }
            all_results.append(result_info)

            # 综合评分
            score = icp_result.fitness - norm_dist[idx] * 0.05

            if score > best_fitness - best_rmse * 0.001:
                if score > best_fitness or (abs(score - best_fitness) < 0.001 and icp_result.inlier_rmse < best_rmse):
                    best_fitness = score
                    best_rmse = icp_result.inlier_rmse
                    best_idx = idx
                    best_transform = icp_result.transformation

            if (idx + 1) % 20 == 0:
                print(f"    已完成 {idx+1}/{n_models} 个模型")

        except Exception as e:
            all_results.append({
                "rank": rank,
                "index": idx,
                "path": db["paths"][idx],
                "fitness": 0.0,
                "rmse": float("inf"),
            })

    # 回退机制：如果最佳fitness太低，使用粗筛结果
    if best_fitness < 0.01 and coarse_distances is not None:
        print(f"  配准质量低，回退到粗筛结果")
        best_idx = np.argmin(coarse_distances)
        best_fitness = 0.0

    if best_idx == -1:
        if coarse_distances is not None:
            best_idx = np.argmin(coarse_distances)
        else:
            raise RuntimeError("所有模型配准均失败")

    return {
        "best_index": best_idx,
        "best_path": db["paths"][best_idx],
        "fitness": best_fitness if best_idx >= 0 else 0.0,
        "rmse": best_rmse if best_idx >= 0 else float("inf"),
        "transformation": best_transform,
        "all_results": sorted(all_results, key=lambda x: (-x["fitness"], x["rmse"])),
    }


def fine_match(scan_down, scan_fpfh, candidates, db, voxel_size):
    """接口保留"""
    if len(candidates) < len(db["paths"]):
        return fine_match_candidates(scan_down, scan_fpfh, candidates, db, voxel_size)
    else:
        return fine_match_all(scan_down, scan_fpfh, db, voxel_size)


def fine_match_candidates(scan_down, scan_fpfh, candidates, db, voxel_size):
    """对粗筛候选进行精匹配"""
    best_idx = -1
    best_fitness = -1.0
    best_rmse = float("inf")
    best_transform = None
    all_results = []

    # 记录粗筛排名，用于综合评分
    coarse_rank_map = {idx: rank for rank, idx in enumerate(candidates)}

    for rank, idx in enumerate(candidates):
        cad_down_np = db["downsampled"][idx]
        cad_down = o3d.geometry.PointCloud()
        cad_down.points = o3d.utility.Vector3dVector(cad_down_np)
        cad_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )

        cad_fpfh_np = db["fpfh"][idx]
        cad_fpfh = o3d.pipelines.registration.Feature()
        cad_fpfh.data = cad_fpfh_np.astype(np.float64)

        try:
            fgr_result = fast_global_registration(scan_down, cad_down, scan_fpfh, cad_fpfh, voxel_size)
            icp_result = refine_with_icp(scan_down, cad_down, fgr_result.transformation, voxel_size)

            # 综合评分：fitness权重高，但也要考虑粗筛排名
            coarse_rank = coarse_rank_map[idx]
            coarse_weight = 0.1 * (1 - coarse_rank / len(candidates))  # 粗筛排名越高，权重越大

            combined_score = icp_result.fitness + coarse_weight

            result_info = {
                "rank": rank,
                "index": idx,
                "path": db["paths"][idx],
                "fitness": icp_result.fitness,
                "rmse": icp_result.inlier_rmse,
                "combined_score": combined_score,
                "coarse_rank": coarse_rank,
                "transformation": icp_result.transformation,
            }
            all_results.append(result_info)

            # 使用综合评分选择最佳匹配
            if combined_score > best_fitness or (abs(combined_score - best_fitness) < 0.001 and icp_result.inlier_rmse < best_rmse):
                best_fitness = combined_score
                best_rmse = icp_result.inlier_rmse
                best_idx = idx
                best_transform = icp_result.transformation

        except Exception as e:
            all_results.append({
                "rank": rank,
                "index": idx,
                "path": db["paths"][idx],
                "fitness": 0.0,
                "rmse": float("inf"),
                "combined_score": 0.0,
            })

    if best_idx == -1:
        best_idx = candidates[0]

    # 回退机制：当最佳和次佳fitness非常接近时，参考粗筛结果
    if len(all_results) >= 2:
        sorted_results = sorted(all_results, key=lambda x: (-x["fitness"], x["rmse"]))
        top1_fitness = sorted_results[0]["fitness"]
        top2_fitness = sorted_results[1]["fitness"]

        # 如果fitness差距小于0.005，优先选择粗筛排名更靠前的
        if top1_fitness - top2_fitness < 0.005:
            if sorted_results[1]["coarse_rank"] < sorted_results[0]["coarse_rank"]:
                # 交换选择
                best_idx = sorted_results[1]["index"]
                best_fitness = sorted_results[1]["fitness"]
                best_rmse = sorted_results[1]["rmse"]
                best_transform = sorted_results[1].get("transformation")

    return {
        "best_index": best_idx,
        "best_path": db["paths"][best_idx],
        "fitness": best_fitness if best_idx >= 0 else 0.0,
        "rmse": best_rmse if best_idx >= 0 else float("inf"),
        "transformation": best_transform,
        "all_results": sorted(all_results, key=lambda x: (-x["fitness"], x["rmse"])),
    }
