# Scan Match CAD

从300个CAD模型中快速匹配扫描数据的最佳模型，匹配耗时 < 3秒。

## 项目简介

本项目实现了一套**两阶段匹配法**，通过全局描述符粗筛 + 精细配准，高效地从300个CAD模型中找到与扫描点云最佳匹配的模型。

**匹配流程：**
```
扫描点云 → 预处理 → 提取全局描述符 → 粗筛(Top-K) → 精配准(FGR+ICP) → 输出最佳匹配
```

## 性能估计

| 阶段           | 方法                     | 预估耗时     |
|----------------|--------------------------|--------------|
| 粗筛           | FAISS 近似最近邻搜索     | < 1 ms       |
| 精匹配（单个） | FGR + ICP                | 100~200 ms   |
| 精匹配（10个） | FGR + ICP × Top-10       | 1~2 s        |
| **总计**       | 粗筛 + 精匹配            | **< 3 s** ✅ |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 离线构建特征数据库（一次性）

```bash
python offline_preprocess.py --cad_dir ./cad_models --output feature_db.pkl
```

### 3. 运行匹配

```bash
python main_pipeline.py --scan ./scan_data/test_scan.ply --db feature_db.pkl
```

### 4. 使用测试数据验证

```bash
# 生成测试数据（300个模型 + 1个扫描文件）
python generate_test_data.py --output_dir test_data --n_models 300

# 构建测试数据库
python offline_preprocess.py --cad_dir test_data/cad_models --output feature_db.pkl

# 运行匹配验证
python main_pipeline.py --scan test_data/scan_data/test_scan.ply --db feature_db.pkl
```

## 技术方案

### 核心依赖
- **[Open3D](http://www.open3d.org/)**: 点云处理、FPFH特征提取、Fast Global Registration、ICP
- **[FAISS](https://github.com/facebookresearch/faiss)**: 高维向量快速近似最近邻搜索
- **[NumPy](https://numpy.org/)**: 数值计算

### 两阶段匹配详解

**阶段一：粗筛（< 1ms）**
- 对所有CAD模型预提取66维全局描述符（FPFH均值 + 标准差）
- 使用FAISS IndexFlatL2 暴力L2搜索（300个模型时速度极快）
- 返回Top-K个候选（默认K=10）

**阶段二：精匹配（1~2秒）**
- 对每个候选执行 Fast Global Registration（FGR）获取初始变换
- 再用 Point-to-Plane ICP 精细配准
- 按 fitness（内点比例）和 RMSE 综合评分，选出最佳匹配

### 全局描述符构建
```
FPFH特征矩阵 (33×N)
    → 按特征维度计算均值 (33维)
    → 按特征维度计算标准差 (33维)
    → 拼接得到 66维全局描述符
```

## 项目结构

```
scan_match_cad/
├── requirements.txt          # 依赖包
├── offline_preprocess.py     # 离线预处理，构建特征数据库
├── coarse_matching.py        # 粗筛模块（FAISS快速检索）
├── fine_matching.py          # 精匹配模块（FGR + ICP）
├── main_pipeline.py          # 主流程整合
├── generate_test_data.py     # 测试数据生成工具
└── README.md
```

## 性能优化建议

- **增大 `VOXEL_SIZE`**：降低点云密度，加快FGR/ICP速度（精度略降）
- **减少 `top_k`**：减少精匹配候选数量（如 top_k=5）
- **并行精匹配**：使用 `concurrent.futures` 并行处理Top-K候选
- **模型数量 >10000**：切换FAISS索引为 `IndexIVFFlat` 或 `IndexHNSWFlat`

## License

MIT
