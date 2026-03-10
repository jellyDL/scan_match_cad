# Scan Match CAD

从300个CAD模型中快速匹配扫描数据的最佳模型


| 文件                    | 说明                                      |
| ----------------------- | ----------------------------------------- |
| `requirements.txt`      | 依赖清单 (open3d, numpy, faiss-cpu)       |
| `offline_preprocess.py` | 离线预处理：构建300个CAD模型的特征数据库  |
| `coarse_matching.py`    | 粗筛模块：FAISS全局描述符快速检索 (< 1ms) |
| `fine_matching.py`      | 精匹配模块：FGR + ICP 配准                |
| `main_pipeline.py`      | 主流程入口：两阶段匹配完整管线            |
| `generate_test_data.py` | 测试数据生成工具（300个模拟模型）         |
| `.gitignore`            | 忽略数据文件和缓存                        |
| `README.md`             | 完整项目文档                              |
