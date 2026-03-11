# Scan Match CAD

从CAD模型库中匹配扫描数据的最佳模型

## 性能

- **准确率**: ~89% (88/99)
- **匹配耗时**: ~5秒（含99个模型全配准）

## 使用方法

### 第一步：构建特征数据库
```bash
python offline_preprocess.py --cad_dir /path/to/CAD_STL --output feature_db.pkl
```

### 第二步：运行匹配
```bash
python main_pipeline.py --scan /path/to/scan.stl --db feature_db.pkl
```

## 文件说明

| 文件 | 说明 |
| ---- | ---- |
| `requirements.txt` | 依赖清单 |
| `offline_preprocess.py` | 离线预处理：构建CAD模型特征数据库 |
| `coarse_matching.py` | 粗筛模块：全局描述符快速检索 |
| `fine_matching.py` | 精匹配模块：FGR + ICP 配准 |
| `main_pipeline.py` | 主流程入口 |
| `test_accuracy.py` | 准确率测试脚本 |

## 特征配置

- 体素大小: 0.5mm
- 采样点数: 15000
- 特征维度: 245维 (FPFH统计特征 + 几何特征)
