import open3d as o3d
import numpy as np
import os
import sys

# 添加项目路径以导入匹配模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from offline_preprocess import preprocess_and_extract_features, VOXEL_SIZE
from coarse_matching import CoarseMatcher
from fine_matching import fine_match


class DualViewportVisualizer:
    def __init__(self, mesh1, mesh2, transformation=None):
        self.mesh1_path = mesh1
        self.mesh2_path = mesh2
        self.transformation = transformation  # 精匹配RT变换矩阵 (4x4)
        self.should_exit = False

    def run(self):
        """使用两个独立窗口，各自独立旋转，带按键回调"""
        # 调试信息
        print(f"\n[DEBUG] transformation is: {type(self.transformation)}")
        if self.transformation is not None:
            print(f"[DEBUG] transformation shape: {self.transformation.shape}")
            print(f"[DEBUG] transformation:\n{self.transformation}")
        else:
            print("[DEBUG] transformation is None!")

        # 加载网格
        self.mesh1 = o3d.io.read_triangle_mesh(self.mesh1_path)
        self.mesh1.compute_vertex_normals()
        self.mesh1.paint_uniform_color([0.8, 0.3, 0.3])  # 红色

        self.mesh2 = o3d.io.read_triangle_mesh(self.mesh2_path)
        self.mesh2.compute_vertex_normals()
        self.mesh2.paint_uniform_color([0.3, 0.5, 0.8])  # 蓝色

        print("=" * 50)
        print("三窗口网格查看器 (独立旋转)")
        print("=" * 50)
        print("操作说明：")
        print("- 左键拖动：旋转网格（每个窗口独立）")
        print("- 滚轮：缩放")
        print("- 右键拖动：平移")
        print("- 按 'N' 键：退出程序")
        print("- 按 'T' 键：将左侧文件重命名为右侧文件名")
        print("- 窗口1：扫描数据 (红色)")
        print("- 窗口2：CAD模型 (蓝色)")
        if self.transformation is not None:
            print("- 窗口3：精匹配叠加 (红色scan+蓝色CAD)")
        print("=" * 50)

        # 创建带按键回调的可视化器
        self.vis1 = o3d.visualization.VisualizerWithKeyCallback()
        self.vis2 = o3d.visualization.VisualizerWithKeyCallback()

        # 如果有精匹配变换矩阵，创建第三个窗口显示叠加效果
        if self.transformation is not None:
            self.vis3 = o3d.visualization.VisualizerWithKeyCallback()
            # 对mesh1应用RT变换
            self.mesh1_transformed = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(np.asarray(self.mesh1.vertices)),
                o3d.utility.Vector3iVector(np.asarray(self.mesh1.triangles))
            )
            self.mesh1_transformed.compute_vertex_normals()
            self.mesh1_transformed.transform(self.transformation)
            self.mesh1_transformed.paint_uniform_color([0.9, 0.5, 0.5])  # 淡红色

        # 窗口1 - 扫描数据
        self.vis1.create_window(
            window_name="Viewport 1 - 扫描数据 (SCAN)",
            width=800, height=600, left=50, top=50
        )
        self.vis1.add_geometry(self.mesh1)

        # 设置渲染选项
        opt1 = self.vis1.get_render_option()
        opt1.background_color = np.array([0.1, 0.1, 0.12])
        opt1.show_coordinate_frame = True
        ctr1 = self.vis1.get_view_control()
        ctr1.set_zoom(0.8)

        # 注册按键回调
        self.vis1.register_key_callback(ord('N'), self.on_exit)
        self.vis1.register_key_callback(ord('n'), self.on_exit)
        self.vis1.register_key_callback(ord('T'), self.on_rename)
        self.vis1.register_key_callback(ord('t'), self.on_rename)

        # 窗口2 - CAD模型
        self.vis2.create_window(
            window_name="Viewport 2 - CAD模型 (CAD)",
            width=800, height=600, left=900, top=50
        )
        self.vis2.add_geometry(self.mesh2)

        # 设置渲染选项
        opt2 = self.vis2.get_render_option()
        opt2.background_color = np.array([0.12, 0.12, 0.1])
        opt2.show_coordinate_frame = True
        ctr2 = self.vis2.get_view_control()
        ctr2.set_zoom(0.8)

        # 注册按键回调
        self.vis2.register_key_callback(ord('N'), self.on_exit)
        self.vis2.register_key_callback(ord('n'), self.on_exit)
        self.vis2.register_key_callback(ord('T'), self.on_rename)
        self.vis2.register_key_callback(ord('t'), self.on_rename)

        # 窗口3 - 精匹配叠加显示
        if self.transformation is not None:
            self.vis3.create_window(
                window_name="Viewport 3 - 精匹配叠加 (SCAN+CAD)",
                width=800, height=600, left=50, top=700
            )

            # 为窗口3创建网格副本，使用淡色以便区分
            mesh2_v3 = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(np.asarray(self.mesh2.vertices)),
                o3d.utility.Vector3iVector(np.asarray(self.mesh2.triangles))
            )
            mesh2_v3.compute_vertex_normals()
            mesh2_v3.paint_uniform_color([0.5, 0.6, 0.9])  # 淡蓝色

            # 叠加显示：变换后的scan（淡红色）+ CAD（淡蓝色）
            self.vis3.add_geometry(self.mesh1_transformed)
            self.vis3.add_geometry(mesh2_v3)

            # 设置渲染选项
            opt3 = self.vis3.get_render_option()
            opt3.background_color = np.array([0.1, 0.12, 0.1])
            opt3.show_coordinate_frame = True
            opt3.mesh_show_back_face = True  # 显示背面
            ctr3 = self.vis3.get_view_control()
            ctr3.set_zoom(0.8)

            # 注册按键回调
            self.vis3.register_key_callback(ord('N'), self.on_exit)
            self.vis3.register_key_callback(ord('n'), self.on_exit)

        # 单线程轮询渲染窗口
        while not self.should_exit:
            # 渲染窗口1
            if not self.vis1.poll_events():
                self.should_exit = True
                break
            self.vis1.update_renderer()

            # 渲染窗口2
            if not self.vis2.poll_events():
                self.should_exit = True
                break
            self.vis2.update_renderer()

            # 渲染窗口3（精匹配叠加）
            if self.transformation is not None:
                if not self.vis3.poll_events():
                    self.should_exit = True
                    break
                self.vis3.update_renderer()

        # 销毁窗口
        self.vis1.destroy_window()
        self.vis2.destroy_window()
        if self.transformation is not None:
            self.vis3.destroy_window()
        print("程序已退出")

    def on_exit(self, vis):
        """N键退出回调"""
        print("\n[N] 退出程序...")
        self.should_exit = True
        return False

    # def on_rename(self, vis):
    #     """T键重命名回调 - 将左侧文件重命名为右侧文件名"""
    #     print(f"\n[T] 键按下 - 重命名文件")
    #     # 获取右侧文件（CAD）的文件名
    #     cad_filename = os.path.basename(self.mesh2_path)
    #     # 构建新路径：左侧文件所在目录 + 右侧文件名
    #     dir_path = os.path.dirname(self.mesh1_path)
    #     new_path = os.path.join(dir_path, cad_filename)

    #     print(f"  原文件: {self.mesh1_path}")
    #     print(f"  新文件: {new_path}")

    #     try:
    #         if self.mesh1_path != new_path:
    #             os.rename(self.mesh1_path, new_path)
    #             print("✓ 重命名成功!")
    #             # 更新路径
    #             self.mesh1_path = new_path
    #         else:
    #             print("  文件名相同，无需重命名")
    #     except Exception as e:
    #         print(f"✗ 重命名失败: {e}")
    #     return True

    def on_rename(self, vis):
        """T键重命名回调 - 将左侧文件重命名为右侧文件名"""
        print(f"\n[T] 键按下 - 重命名文件")
        # 获取右侧文件（CAD）的文件名
        scan_filename = os.path.basename(self.mesh1_path)
        # 构建新路径：左侧文件所在目录 + 右侧文件名
        dir_path = os.path.dirname(self.mesh2_path)
        new_path = os.path.join(dir_path, scan_filename)

        print(f"  原文件: {self.mesh2_path}")
        print(f"  新文件: {new_path}")

        try:
            if self.mesh2_path != new_path:
                os.rename(self.mesh2_path, new_path)
                print("✓ 重命名成功!")
                # 更新路径
                self.mesh2_path = new_path
            else:
                print("  文件名相同，无需重命名")
        except Exception as e:
            print(f"✗ 重命名失败: {e}")
        return True


def run_matching_for_visualization(scan_path, db_path="feature_db_300.pkl", top_k=99):
    """运行匹配流程获取变换矩阵"""
    import pickle

    # 加载数据库
    with open(db_path, "rb") as f:
        db = pickle.load(f)
    voxel_size = db.get("voxel_size", VOXEL_SIZE)
    print(f"数据库加载完成: {len(db['paths'])} 个模型")

    # 预处理扫描数据
    print(f"预处理扫描数据: {scan_path}")
    scan_down, scan_fpfh, scan_global = preprocess_and_extract_features(scan_path, voxel_size)
    print(f"  降采样点数: {len(scan_down.points)}")

    # 粗筛
    matcher = CoarseMatcher(db["global_matrix"])
    candidates, _ = matcher.search(scan_global, top_k=top_k)
    print(f"粗筛完成, Top-{top_k}候选")

    # 精匹配
    result = fine_match(scan_down, scan_fpfh, candidates, db, voxel_size)
    print(f"精匹配完成: {os.path.basename(result['best_path'])}")
    print(f"  Fitness: {result['fitness']:.6f}")
    print(f"  RMSE: {result['rmse']:.6f}")

    return result


# 运行程序
if __name__ == "__main__":
    # 扫描数据和CAD模型路径
    scan_path = "/Users/jelly/Desktop/Crown_Pair_Dataset_2026_0105/SCAN/0026-0903e0026a-3m2-17-16-waxup_slm_cad.stl"

    # 运行匹配获取变换矩阵
    print("=" * 60)
    print("运行匹配流程获取变换矩阵...")
    result = run_matching_for_visualization(scan_path)

    mesh1 = scan_path
    mesh2 = result["best_path"]
    transformation = result["transformation"]

    # 如果变换矩阵为None，使用单位矩阵
    if transformation is None:
        print("\n警告: 精匹配未返回变换矩阵，使用单位矩阵")
        transformation = np.eye(4)

    print(f"\n最佳匹配: {os.path.basename(mesh2)}")
    print(f"变换矩阵:\n{transformation}")

    # 创建可视化器（传入变换矩阵以显示第三个窗口）
    visualizer = DualViewportVisualizer(mesh1, mesh2, transformation=transformation)
    visualizer.run()
