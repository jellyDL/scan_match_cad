import os
from main_pipeline import run_matching
from visual import DualViewportVisualizer

def evaluate_accuracy(db_path="feature_db.pkl", top_k=99, scan_dir=None, verbose=False, vis=False):
    
    scan_files = sorted(
        [f for f in os.listdir(scan_dir) if f.lower().endswith('.stl')]
    )
    
    match_num = 0
    for iter, scan_file in enumerate(scan_files):
        scan_path = os.path.join(scan_dir, scan_file)
        print(f"\n测试 {iter+1}/{len(scan_files)}: {scan_file}")
        result = run_matching(scan_path, db_path, top_k, verbose)
        if result["best_path"].split("/")[-1] == scan_file:
            print(f"【 Match Success! 】Fitness: {result['fitness']:.4f}")
            match_num += 1
        else:
            print("Match Failed!")
            print(f"  最佳匹配: {result['best_path'].split('/')[-1]}, Fitness: {result['fitness']:.4f}")
            
        if vis:
            scan_file = os.path.join(scan_dir, scan_file)
            cad_file = result["best_path"]
            print(f"可视化配准结果: {scan_file} <-> {cad_file}")
            visualizer = DualViewportVisualizer(scan_file, cad_file)
            visualizer.run()
            
            
            
    print(f"\n最终准确率: {match_num/len(scan_files)*100:.2f}%")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="匹配准确率测试")
    parser.add_argument(
        "--scan_dir", type=str, default="/Users/jelly/Desktop/Crown_Pair_Dataset_2026_03_05/SCAN", help="扫描数据文件路径)"
    )
    parser.add_argument(
        "--db", type=str, default="feature_db.pkl", help="特征数据库路径"
    )
    parser.add_argument(
        "--top_k", type=int, default=200, help="粗筛候选数量"
    )
    parser.add_argument(
        "--verbose", type=bool, default=False, help="详细输出模式"
    )
    parser.add_argument(
        "--vis", type=bool, default=False, help="匹配可视化"
    )
    args = parser.parse_args()

    evaluate_accuracy(args.db, args.top_k, args.scan_dir, args.verbose, args.vis)

if __name__ == "__main__":
    main()
