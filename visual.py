import open3d as o3d
import numpy as np
import os


class DualViewportVisualizer:
    def __init__(self, mesh1, mesh2):
        self.mesh1_path = mesh1
        self.mesh2_path = mesh2
        self.should_exit = False

    def run(self):
        """使用两个独立窗口，各自独立旋转，带按键回调"""
        # 加载网格
        self.mesh1 = o3d.io.read_triangle_mesh(self.mesh1_path)
        self.mesh1.compute_vertex_normals()
        self.mesh1.paint_uniform_color([0.8, 0.3, 0.3])  # 红色

        self.mesh2 = o3d.io.read_triangle_mesh(self.mesh2_path)
        self.mesh2.compute_vertex_normals()
        self.mesh2.paint_uniform_color([0.3, 0.5, 0.8])  # 蓝色

        print("=" * 50)
        print("双窗口网格查看器 (独立旋转)")
        print("=" * 50)
        print("操作说明：")
        print("- 左键拖动：旋转网格（每个窗口独立）")
        print("- 滚轮：缩放")
        print("- 右键拖动：平移")
        print("- 按 'N' 键：退出程序")
        print("- 按 'T' 键：将左侧文件重命名为右侧文件名")
        print("=" * 50)

        # 创建带按键回调的可视化器
        self.vis1 = o3d.visualization.VisualizerWithKeyCallback()
        self.vis2 = o3d.visualization.VisualizerWithKeyCallback()

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

        # 单线程轮询渲染两个窗口
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

        # 销毁窗口
        self.vis1.destroy_window()
        self.vis2.destroy_window()
        print("程序已退出")

    def on_exit(self, vis):
        """N键退出回调"""
        print("\n[N] 退出程序...")
        self.should_exit = True
        return False

    def on_rename(self, vis):
        """T键重命名回调 - 将左侧文件重命名为右侧文件名"""
        print(f"\n[T] 键按下 - 重命名文件")
        # 获取右侧文件（CAD）的文件名
        cad_filename = os.path.basename(self.mesh2_path)
        # 构建新路径：左侧文件所在目录 + 右侧文件名
        dir_path = os.path.dirname(self.mesh1_path)
        new_path = os.path.join(dir_path, cad_filename)

        print(f"  原文件: {self.mesh1_path}")
        print(f"  新文件: {new_path}")

        try:
            if self.mesh1_path != new_path:
                os.rename(self.mesh1_path, new_path)
                print("✓ 重命名成功!")
                # 更新路径
                self.mesh1_path = new_path
            else:
                print("  文件名相同，无需重命名")
        except Exception as e:
            print(f"✗ 重命名失败: {e}")
        return True


# 运行程序
if __name__ == "__main__":
    mesh1 = "/Users/jelly/Desktop/Crown_Pair_Dataset_2026_0105/SCAN/0026-0903e0026a-3m2-17-16-waxup_slm_cad.stl"
    mesh2 = "/Users/jelly/Desktop/Crown_Pair_Dataset_2026_0105/CAD/0026-0903e0026a-3m2-17-16-waxup_slm_cad.stl"
    visualizer = DualViewportVisualizer(mesh1, mesh2)
    visualizer.run()
