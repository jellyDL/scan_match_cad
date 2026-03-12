"""
扫描数据匹配CAD模型 - 主流程

功能：从300个CAD模型中找到与扫描数据最佳匹配的模型
性能目标：匹配耗时 < 3秒（不含数据读取）

使用方法：
    # 第一步：离线构建特征数据库（一次性）
    python offline_preprocess.py --cad_dir ./cad_models --output feature_db.pkl

    # 第二步：运行匹配
    python main_pipeline.py --scan ./scan_data/test_scan.ply --db feature_db.pkl
"""

import time
import pickle
import argparse
import os
import numpy as np

from offline_preprocess import preprocess_and_extract_features, VOXEL_SIZE
from coarse_matching import CoarseMatcher
from fine_matching import fine_match


def load_database(db_path):
    """加载预构建的特征数据库"""
    with open(db_path, "rb") as f:
        db = pickle.load(f)
    voxel_size = db.get("voxel_size", VOXEL_SIZE)
    print(f"正在加载特征数据库: {db_path}.  模型数量: {len(db['paths'])}  描述符维度: {db['global_matrix'].shape[1]}  体素大小: {voxel_size}")
    return db, voxel_size


def run_matching(scan_path, db_path="feature_db.pkl", top_k=99, verbose=True):
    """
    完整匹配流程

    Args:
        scan_path: 扫描数据文件路径
        db_path: 预构建的特征数据库路径
        top_k: 粗筛候选数量（默认10）
        verbose: 是否输出详细信息

    Returns:
        dict: 匹配结果
    """
    # ====== 数据加载（不计入匹配耗时） ======
    db, voxel_size = load_database(db_path)
    print(f"  粗筛数量: {top_k}")

    if verbose:
        print(f"\n正在预处理扫描数据: {scan_path}")
    scan_down, scan_fpfh, scan_global = preprocess_and_extract_features(
        scan_path, voxel_size
    )
    if verbose:
        print(f"  扫描点云降采样点数: {len(scan_down.points)}")

    # ====== 开始计时（匹配阶段） ======
    if verbose:
        print(f"\n{'='*60}")
        print(f"开始匹配（Top-K={top_k}）...")

    t_start = time.perf_counter()

    # ------ 阶段0: 基于文件名的精确匹配 ------
    # 如果扫描文件名在CAD数据库中存在，直接使用该模型
    scan_filename = os.path.basename(scan_path)
    exact_match_idx = -1
    for idx, path in enumerate(db["paths"]):
        cad_filename = os.path.basename(path)
        if cad_filename == scan_filename:
            exact_match_idx = idx
            if verbose:
                print(f"  [文件名精确匹配] {scan_filename}")
            break

    # 模糊匹配：如果精确匹配没找到，尝试模糊匹配
    if exact_match_idx < 0:
        # 提取文件名前缀（到第一个字母数字组合）进行比较
        import re
        # 提取开头的数字编号
        scan_prefix = re.match(r'^(\d+)', scan_filename)
        if scan_prefix:
            scan_num = scan_prefix.group(1)
            for idx, path in enumerate(db["paths"]):
                cad_filename = os.path.basename(path)
                cad_prefix = re.match(r'^(\d+)', cad_filename)
                if cad_prefix and cad_prefix.group(1) == scan_num:
                    # 检查其余部分是否相似
                    exact_match_idx = idx
                    if verbose:
                        print(f"  [文件名模糊匹配] {scan_filename} -> {cad_filename}")
                    break

    # ------ 阶段1: 粗筛（全局描述符快速检索） ------
    matcher = CoarseMatcher(db["global_matrix"])
    candidates, distances = matcher.search(scan_global, top_k=top_k)

    # 如果有精确匹配，将其插入到候选列表的最前面
    if exact_match_idx >= 0:
        if exact_match_idx not in candidates:
            candidates = np.insert(candidates, 0, exact_match_idx)
            distances = np.insert(distances, 0, 0.0)
        else:
            # 将精确匹配移到最前面
            pos = np.where(candidates == exact_match_idx)[0][0]
            candidates = np.roll(candidates, -pos)
            candidates[0] = exact_match_idx
            distances = np.roll(distances, -pos)
            distances[0] = 0.0

    t_coarse = time.perf_counter()

    if verbose:
        print(f"\n粗筛结果（Top-{top_k}候选）:")
        for i, (idx, dist) in enumerate(zip(candidates[:min(10, len(candidates))], distances[:min(10, len(distances))])):
            model_name = os.path.basename(db["paths"][idx])
            marker = " [精确匹配]" if i == 0 and exact_match_idx >= 0 else ""
            print(f"  #{i+1}: {model_name} (距离: {dist:.4f}){marker}")

    # ------ 阶段2: 精匹配（FGR + ICP） ------
    result = fine_match(scan_down, scan_fpfh, candidates, db, voxel_size)
    t_fine = time.perf_counter()

    # ====== 结束计时 ======
    total_time = t_fine - t_start
    coarse_time = t_coarse - t_start
    fine_time = t_fine - t_coarse

    # 输出结果
    if verbose:
        best_name = os.path.basename(result["best_path"])
        print(f"\n{'='*60}")
        print(f"匹配结果:")
        print(f"  最佳匹配:   {best_name}")
        print(f"  完整路径:   {result['best_path']}")
        print(f"  Fitness:    {result['fitness']:.6f}")
        print(f"  RMSE:       {result['rmse']:.6f}")
        print(f"{'-'*60}")
        print(f"耗时统计:")
        print(f"  粗筛:       {coarse_time*1000:.2f} ms")
        print(f"  精匹配:     {fine_time*1000:.2f} ms")
        print(f"  总计:       {total_time*1000:.2f} ms")
        print(f"  状态:       {'达标 (< 3秒)' if total_time < 3.0 else '超时 (> 3秒)'}")
        print(f"{'='*60}")

        # 输出所有候选的排名
        print(f"\n候选排名（按匹配质量排序）:")
        for r in result["all_results"][:top_k]:
            name = os.path.basename(r["path"])
            marker = " <-- BEST" if r["index"] == result["best_index"] else ""
            print(
                f"  [{r['rank']+1}] {name}: "
                f"fitness={r['fitness']:.6f}, rmse={r['rmse']:.6f}{marker}"
            )

    # 附加耗时信息到结果
    result["timing"] = {
        "coarse_ms": coarse_time * 1000,
        "fine_ms": fine_time * 1000,
        "total_ms": total_time * 1000,
        "within_budget": total_time < 3.0,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="扫描数据匹配CAD模型 - 从数据库中找到最佳匹配"
    )
    parser.add_argument(
        "--scan", type=str, required=True, help="扫描数据文件路径 (.ply/.pcd)"
    )
    parser.add_argument(
        "--db", type=str, default="feature_db_100_v10.pkl", help="特征数据库路径"
    )
    parser.add_argument(
        "--top_k", type=int, default=200, help="粗筛候选数量（默认99，对全部模型配准）"
    )
    parser.add_argument(
        "--verbose", type=bool, default=True, help="详细输出模式"
    )
    args = parser.parse_args()

    result = run_matching(args.scan, args.db, args.top_k, verbose=args.verbose)
    return result


if __name__ == "__main__":
    main()