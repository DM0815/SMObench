# SMObench 用户指南

完整的安装、配置和使用教程。

---

## 目录

1. [安装](#1-安装)
2. [数据准备](#2-数据准备)
3. [快速上手：Python API](#3-快速上手python-api)
4. [命令行 CLI](#4-命令行-cli)
5. [结果存储与读取（h5ad）](#5-结果存储与读取h5ad)
6. [可视化](#6-可视化)
7. [方法详解](#7-方法详解)
8. [评估指标](#8-评估指标)
9. [环境隔离（Singularity / Conda）](#9-环境隔离singularity--conda)
10. [添加自定义方法](#10-添加自定义方法)
11. [配置文件](#11-配置文件)
12. [常见问题](#12-常见问题)

---

## 1. 安装

### 1.1 基础安装

```bash
git clone https://github.com/your-org/SMObench.git
cd SMObench
pip install -e .
```

这会安装核心依赖（scanpy, numpy, pandas, scikit-learn 等），足以运行 SMOPCA 等不依赖 GPU 的方法。

### 1.2 按方法安装额外依赖

大多数方法需要 PyTorch + PyTorch Geometric：

```bash
# 先装 PyTorch（根据你的 CUDA 版本）
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 装 PyG
pip install smobench[pyg]

# 装特定方法的依赖
pip install smobench[cosmos]      # 额外需要 gudhi, cmcrameri
pip install smobench[spamosaic]   # 额外需要 dgl, hnswlib, annoy
pip install smobench[multigate]   # 需要 tensorflow
```

### 1.3 一键安装所有方法

```bash
pip install smobench[all-methods]
```

### 1.4 可用的 extras

| Extra | 方法 | 关键依赖 |
|-------|------|----------|
| `torch` | 所有 GPU 方法 | torch >= 2.0 |
| `pyg` | SpatialGlue, PRAGA, GraphST 等 | torch-geometric, torch-scatter, torch-sparse |
| `dgl` | SpaMosaic | dgl |
| `cosmos` | COSMOS | gudhi, cmcrameri |
| `multigate` | MultiGATE | tensorflow |
| `smopca` | SMOPCA | （无额外依赖） |
| `all-pyg` | 所有 PyTorch 方法 | 全部 PyG 相关依赖 |
| `all-methods` | 全部 15 个方法 | 包括 dgl, tensorflow 等 |

### 1.5 验证安装

```bash
smobench list methods    # 查看所有方法及其环境状态
smobench list datasets   # 查看所有数据集
```

```python
import smobench
smobench.list_methods()   # 返回 DataFrame
smobench.list_datasets()
```

---

## 2. 数据准备

### 2.1 数据目录结构

SMObench 的数据按如下结构组织：

```
Dataset/
├── withGT/                          # 有 ground truth 标签
│   ├── RNA_ADT/
│   │   ├── Human_Lymph_Nodes/
│   │   │   ├── A1/
│   │   │   │   ├── adata_RNA.h5ad
│   │   │   │   └── adata_ADT.h5ad
│   │   │   └── D1/
│   │   └── Human_Tonsils/
│   └── RNA_ATAC/
│       ├── Mouse_Embryos_S1/
│       └── Mouse_Embryos_S2/
├── woGT/                            # 无 ground truth
│   ├── RNA_ADT/
│   │   ├── Mouse_Spleen/
│   │   └── Mouse_Thymus/
│   └── RNA_ATAC/
│       └── Mouse_Brain/
├── fusionWithGT/                    # 水平整合用融合数据
└── fusionWoGT/
```

### 2.2 设置数据路径

```python
# 方式 1: 函数参数
adata_rna, adata_mod2 = smobench.load_dataset(
    "Human_Lymph_Nodes", "A1",
    data_root="/path/to/Dataset"
)

# 方式 2: 环境变量
import os
os.environ["SMOBENCH_DATA_ROOT"] = "/path/to/Dataset"
```

### 2.3 数据格式

每个 slice 包含两个 h5ad 文件：

```python
import scanpy as sc

adata_rna = sc.read_h5ad("adata_RNA.h5ad")
# adata_rna.X          → 基因表达矩阵
# adata_rna.obsm["spatial"] → 空间坐标
# adata_rna.obs["Spatial_Label"] → ground truth 标签（仅 withGT 数据集）

adata_mod2 = sc.read_h5ad("adata_ADT.h5ad")  # 或 adata_ATAC.h5ad
```

### 2.4 内置数据集一览

| 数据集 | 模态 | Ground Truth | Slices | 聚类数 |
|--------|------|-------------|--------|--------|
| Human_Lymph_Nodes | RNA+ADT | Yes | A1, D1 | 10 |
| Human_Tonsils | RNA+ADT | Yes | S1, S2, S3 | 4 |
| Mouse_Embryos_S1 | RNA+ATAC | Yes | E11, E13, E15, E18 | 14 |
| Mouse_Embryos_S2 | RNA+ATAC | Yes | E11, E13, E15, E18 | 14 |
| Mouse_Spleen | RNA+ADT | No | Spleen1, Spleen2 | 5 |
| Mouse_Thymus | RNA+ADT | No | Thymus1-4 | 5 |
| Mouse_Brain | RNA+ATAC | No | ATAC, H3K27ac, H3K27me3, H3K4me3 | 15 |

---

## 3. 快速上手：Python API

### 3.1 一行运行 benchmark

```python
import smobench

results = smobench.benchmark(
    dataset="Human_Lymph_Nodes",       # 数据集名，或 "all"
    methods=["SpatialGlue", "PRAGA"],  # 方法列表，或 "all"
    task="vertical",                    # "vertical" / "horizontal" / "mosaic" / "all"
    clustering=["leiden", "kmeans"],    # 聚类方法
    device="cuda:0",                   # GPU 设备
    seed=42,
    data_root="/path/to/Dataset",
    save_dir="./results",              # 保存 h5ad 结果
)
```

### 3.2 查看结果

```python
# 返回 BenchmarkResult 对象
print(results)
# → BenchmarkResult(2 methods × 1 datasets, 8 records)

# 转为 DataFrame
df = results.to_dataframe()
print(df)

# 按方法汇总指标
results.summary()

# 按某个指标排名
results.ranking("BioC_Score")
```

### 3.3 单独运行某个方法

如果只想运行一个方法、自己控制流程：

```python
from smobench.data import load_dataset
from smobench.pipeline._isolation import subprocess_integrate
from smobench.clustering import cluster
from smobench.metrics.evaluate import evaluate

# 加载数据
adata_rna, adata_mod2 = load_dataset("Human_Lymph_Nodes", "A1")

# 整合（自动选择环境）
embedding = subprocess_integrate(
    "SpatialGlue", adata_rna, adata_mod2,
    device="cuda:0", seed=42, modality="ADT",
)

# 存储 embedding
adata_rna.obsm["SpatialGlue"] = embedding

# 聚类
cluster(adata_rna, method="leiden", n_clusters=10,
        embedding_key="SpatialGlue", key_added="SpatialGlue_leiden")

# 评估
scores = evaluate(
    adata_rna,
    embedding_key="SpatialGlue",
    cluster_key="SpatialGlue_leiden",
    label_key="Spatial_Label",
    has_gt=True,
)
print(scores)
# → {'ARI': 0.85, 'NMI': 0.72, 'cASW': 0.45, 'Moran_I': 0.12, ...}
```

### 3.4 批量运行所有方法 × 所有数据集

```python
results = smobench.benchmark(
    dataset="all",
    methods="all",
    task="vertical",
    save_dir="./results",
    device="cuda:0",
)

# 保存汇总 CSV（h5ad 已在 save_dir 中自动保存）
results.save("benchmark_summary.csv")
```

---

## 4. 命令行 CLI

### 4.1 基本用法

```bash
# 运行 benchmark
smobench run \
    --dataset Human_Lymph_Nodes \
    --methods SpatialGlue,PRAGA,SMOPCA \
    --task vertical \
    --device cuda:0 \
    --save-dir ./results

# 运行所有方法 × 所有数据集
smobench run --dataset all --methods all --task all --save-dir ./results
```

### 4.2 查看方法和数据集

```bash
smobench list methods     # 列出所有方法、支持的任务、模态、环境状态
smobench list datasets    # 列出所有数据集、slices、聚类数
```

### 4.3 使用配置文件

```bash
# 生成默认配置
smobench init
# → 生成 smobench_config.yaml

# 用配置文件运行
smobench run --config smobench_config.yaml
```

配置文件示例：

```yaml
dataset: all
methods: [SpatialGlue, PRAGA, SMOPCA]
task: vertical
clustering: [leiden, kmeans]
device: cuda:0
seed: 42
data_root: /path/to/Dataset
save_dir: ./results
n_jobs: 1
```

### 4.4 环境配置

```bash
# 为指定方法配置 conda 环境
smobench setup --methods SpatialGlue,PRAGA --backend conda

# 为所有方法配置 singularity 环境
smobench setup --methods all --backend singularity
```

### 4.5 生成图表

```bash
smobench plot heatmap --input results.csv --output figures/
smobench plot radar --input results.csv
smobench plot scatter --input results.csv
```

---

## 5. 结果存储与读取（h5ad）

### 5.1 存储约定

运行 benchmark 时设置 `save_dir`，结果自动存为 h5ad 文件：

```
results/
└── vertical/
    └── Human_Lymph_Nodes/
        ├── A1/
        │   └── adata_integrated.h5ad
        └── D1/
            └── adata_integrated.h5ad
```

每个 `adata_integrated.h5ad` 包含：

| 位置 | Key | 内容 |
|------|-----|------|
| `adata.obsm['{Method}']` | 如 `SpatialGlue` | 联合 embedding (n_cells × n_dims) |
| `adata.obs['{Method}_{clustering}']` | 如 `SpatialGlue_leiden` | 聚类标签 |
| `adata.uns['{Method}_train_time']` | 如 `SpatialGlue_train_time` | 训练时间（秒） |
| `adata.uns['methods_completed']` | — | 已完成方法列表 |
| `adata.uns['smobench_version']` | — | 包版本号 |

### 5.2 读取结果

```python
from smobench.io import load_integrated, get_embedding, list_methods_in_file

# 查看文件中有哪些方法
methods = list_methods_in_file("results/vertical/Human_Lymph_Nodes/A1/adata_integrated.h5ad")
print(methods)  # ['SpatialGlue', 'PRAGA', 'SMOPCA', ...]

# 加载
adata = load_integrated("results/vertical/Human_Lymph_Nodes/A1/adata_integrated.h5ad")
print(adata.obsm.keys())  # 所有 embedding

# 取某个方法的 embedding
emb = get_embedding(adata, "SpatialGlue")
print(emb.shape)  # (n_cells, n_dims)
```

### 5.3 与 scanpy 联合使用

h5ad 文件天然兼容 scanpy，无需转换：

```python
import scanpy as sc

adata = sc.read_h5ad("results/vertical/Human_Lymph_Nodes/A1/adata_integrated.h5ad")

# 用某个方法的 embedding 做 UMAP
sc.pp.neighbors(adata, use_rep="SpatialGlue")
sc.tl.umap(adata)
sc.pl.umap(adata, color=["SpatialGlue_leiden", "Spatial_Label"])

# 对比不同方法
for method in ["SpatialGlue", "PRAGA", "SMOPCA"]:
    sc.pp.neighbors(adata, use_rep=method)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=f"{method}_leiden", title=method)
```

### 5.4 追加写入

多次运行时，`save_embedding()` 会自动追加到已有文件中，不会覆盖之前的方法结果：

```python
from smobench.io import save_embedding

# 第一次写入 SpatialGlue
save_embedding(adata_rna, "SpatialGlue", emb_sg, "adata_integrated.h5ad", train_time=42.5)

# 第二次写入 PRAGA — 追加到同一个文件
save_embedding(adata_rna, "PRAGA", emb_praga, "adata_integrated.h5ad", train_time=38.2)
```

---

## 6. 可视化

### 6.1 从 BenchmarkResult 直接画图

```python
results = smobench.benchmark(...)

# 热力图：方法 × 数据集指标矩阵
results.plot.heatmap()

# 散点图：SC vs BioC
results.plot.scatter()

# 雷达图：多维指标对比
results.plot.radar()

# 运行时间柱状图
results.plot.runtime()

# 排名变化图
results.plot.bump()
```

### 6.2 从 DataFrame 画图

```python
from smobench.plot import heatmap, radar, sc_vs_bioc, runtime_bar

df = results.to_dataframe()

heatmap(df, score_col="BioC_Score", save="figures/heatmap.png")
radar(df, save="figures/radar.png")
sc_vs_bioc(df, save="figures/scatter.png")
runtime_bar(df, save="figures/runtime.png")
```

### 6.3 空间可视化

```python
from smobench.plot import umap_spatial, method_comparison_grid

adata = sc.read_h5ad("adata_integrated.h5ad")

# 单方法 UMAP + 空间图
umap_spatial(adata)

# 多方法对比网格
method_comparison_grid(adata)
```

---

## 7. 方法详解

### 7.1 所有 15 个方法

| 方法 | 类别 | GPU | 环境组 | 关键依赖 |
|------|------|-----|--------|----------|
| **SpatialGlue** | GNN | Yes | torch-pyg | torch, torch-geometric |
| **PRAGA** | GNN | Yes | torch-pyg | torch, torch-geometric, scvi-tools |
| **COSMOS** | Contrastive | Yes | torch-pyg | torch, torch-geometric, gudhi |
| **PRESENT** | GNN | Yes | torch-pyg | torch, torch-geometric |
| **CANDIES** | Mosaic | Yes | torch-pyg | torch, torch-geometric, einops |
| **SpaBalance** | Mosaic | Yes | torch-pyg | torch, torch-geometric |
| **SpaFusion** | Other | Yes | torch-pyg | torch, torch-geometric, munkres |
| **SpaMI** | Contrastive | Yes | torch-pyg | torch, torch-geometric, POT |
| **SpaMV** | VAE | Yes | torch-pyg | torch, torch-geometric, pyro-ppl |
| **SWITCH** | Other | Yes | torch-pyg | torch, torch-geometric |
| **MISO** | Factorization | Yes | torch-pyg | torch, einops |
| **spaMultiVAE** | VAE | Yes | torch-pyg | torch |
| **SpaMosaic** | Mosaic | Yes | spamosaic | torch, dgl, torch-geometric |
| **MultiGATE** | GNN | Yes | multigate | tensorflow |
| **SMOPCA** | Factorization | No | base | numpy, scanpy |

### 7.2 空间坐标使用情况

大多数方法使用 **kNN 图**构建空间邻接关系，不受坐标尺度影响。

需要注意的方法：
- **MultiGATE**: 使用 `Cal_Spatial_Net(rad_cutoff=...)` 基于欧氏距离构建空间图，**rad_cutoff 需要根据坐标系调整**
  - Visium 网格坐标 [0, ~127]：`rad_cutoff=40`（原文设定）
  - 像素坐标 [0, ~2000+]：`rad_cutoff=100~1000`
- **SWITCH**: 基于距离但内部自动估计 radius，无需手动设置
- **spaMultiVAE**: 内部对坐标做 MinMax 归一化，无需担心尺度

### 7.3 任务支持

| 任务 | 说明 | 支持方法 |
|------|------|----------|
| **Vertical** | 同 slice，跨模态整合 | 全部 15 个 |
| **Horizontal** | 同模态，跨 batch 整合 | 全部（通过融合数据） |
| **Mosaic** | 三模态 / 马赛克整合 | SpatialGlue, SpaMosaic, PRAGA, SpaBalance, MISO, PRESENT, SMOPCA, SpaMV |

---

## 8. 评估指标

### 8.1 五个维度

| 维度 | 指标 | 适用场景 |
|------|------|----------|
| **SC** (空间一致性) | Moran's I | 所有任务 |
| **BioC** (生物保守性) | ARI, NMI, cASW, cLISI | 有 GT 数据集 |
| **BVC** (聚类质量) | Silhouette, DBI, CHI | 无 GT 数据集 |
| **BER** (批次效应消除) | kBET, bASW, iLISI, KNN_conn, PCR | 水平/马赛克任务 |
| **CM-GTC** (跨模态拓扑一致性) | CMGTC | 所有任务 |

### 8.2 综合评分公式

```
Vertical (有 GT):   SMObench_V = mean(SC_Score, BioC_Score, CMGTC)
Vertical (无 GT):   SMObench_V = mean(SC_Score, BVC_Score, CMGTC)
Horizontal:         SMObench_H = mean(SC_Score, BVC_Score, BER_Score, CMGTC)
```

### 8.3 自定义评估

```python
from smobench.metrics.evaluate import evaluate, fast, standard, all_metrics

# 快速评估（只算 SC + Silhouette）
scores = fast(adata, embedding_key="SpatialGlue", cluster_key="SpatialGlue_leiden")

# 标准评估（SC + BioC/BVC）
scores = standard(adata, ...)

# 完整评估（所有指标）
scores = all_metrics(adata, ..., label_key="Spatial_Label", batch_key="batch")
```

---

## 9. 环境隔离（Singularity / Conda）

15 个方法的依赖有冲突（PyTorch vs TensorFlow, 不同版本的 PyG/DGL 等）。SMObench 通过子进程隔离自动解决。

### 9.1 解析优先级

调用 `subprocess_integrate()` 时，自动按以下顺序选择运行环境：

1. **当前环境** — 如果依赖已安装，直接在当前进程运行（零开销）
2. **Singularity 容器** — 使用共享 SIF 镜像 + 每方法独立的 pip 前缀
3. **Conda 环境** — `conda run -n smobench_torch python -m smobench._runner ...`

### 9.2 Singularity（推荐用于 HPC）

```bash
# 目录结构
singularity/
├── images/
│   └── smobench_full.sif   # 共享容器（含 PyTorch, scanpy 等）
├── envs/                    # 每方法独立依赖
│   ├── SpatialGlue/         # PYTHONPATH 前缀
│   ├── COSMOS/
│   ├── MultiGATE/
│   └── ...
└── defs/
    └── smobench_full.def    # 容器定义文件
```

环境变量：
```bash
export SMOBENCH_SINGULARITY_PATH=/path/to/singularity   # 自定义 singularity 路径
export SMOBENCH_DATA_ROOT=/path/to/Dataset               # 数据根目录
```

SMObench 会自动在常见 HPC 路径搜索 singularity/apptainer：
- `/app/apps/singularity/*/bin`
- `/usr/local/bin`
- `/opt/singularity/bin`

### 9.3 Conda

```bash
# 自动配置 conda 环境
smobench setup --methods all --backend conda

# 手动创建
conda create -n smobench_torch python=3.10 pytorch torchvision pytorch-geometric -c pytorch -c pyg
conda create -n smobench_multigate python=3.10 tensorflow
```

### 9.4 工作原理

子进程隔离的数据流：

```
主进程                           子进程（容器/conda）
  │                                │
  ├─ 写入 temp h5ad ──────────────→│
  │   (adata_rna, adata_mod2)      │
  │                                ├─ 加载 h5ad
  │                                ├─ 运行 method.integrate()
  │                                ├─ 保存 embedding.npy
  │                                │
  │←── 读取 embedding.npy ─────────┤
  │                                │
  ├─ 聚类 + 评估（主进程）          │
  ├─ 存入 h5ad                     │
  └─ 清理临时文件                   │
```

---

## 10. 添加自定义方法

### 10.1 装饰器方式（最简单）

```python
import smobench

@smobench.register_method("MyMethod", tasks=["vertical", "horizontal"])
def my_method(adata_rna, adata_mod2, device="cuda:0", seed=42, **kwargs):
    # 你的整合逻辑
    embedding = ...
    return embedding  # np.ndarray, shape (n_cells, n_dims)

# 立即可用
results = smobench.benchmark("Human_Lymph_Nodes", methods=["MyMethod"])
```

### 10.2 类继承方式（更多控制）

```python
from smobench.methods import BaseMethod

class MyMethod(BaseMethod):
    name = "MyMethod"
    tasks = ["vertical", "horizontal"]
    modalities = ["RNA+ADT", "RNA+ATAC"]
    requires_gpu = True
    paper = "Author et al., Nature 2025"
    url = "https://github.com/..."

    def integrate(self, adata_rna, adata_mod2, device="cuda:0", seed=42, **kwargs):
        # 你的整合逻辑
        return embedding  # np.ndarray

    def check_deps(self) -> bool:
        """检查依赖是否可用"""
        try:
            import my_package
            return True
        except ImportError:
            return False
```

### 10.3 Entry Point 方式（独立包，无需 fork）

在你自己包的 `pyproject.toml` 中：

```toml
[project.entry-points."smobench.methods"]
MyMethod = "my_package.integration:MyMethodClass"
```

安装后 SMObench 自动发现：

```bash
pip install my-package
smobench list methods  # MyMethod 会出现在列表中
```

---

## 11. 配置文件

### 11.1 生成默认配置

```bash
smobench init
```

生成 `smobench_config.yaml`：

```yaml
# SMObench 配置文件
dataset: all                        # 数据集名，或 "all"
methods: all                        # 方法列表，或 "all"
task: vertical                      # vertical / horizontal / mosaic / all
clustering:
  - leiden
  - kmeans
device: cuda:0
seed: 42
data_root: /path/to/Dataset
save_dir: ./results
n_jobs: 1                           # 并行任务数
```

### 11.2 使用配置文件

```bash
smobench run --config smobench_config.yaml

# CLI 参数覆盖配置文件
smobench run --config config.yaml --device cpu --methods SMOPCA
```

### 11.3 可调参数

```python
import smobench

# 修改全局常量
smobench.config.N_NEIGHBORS = 30          # 邻居数（默认 20）
smobench.config.DEFAULT_RESOLUTION = 1.5  # Leiden 分辨率（默认 1.0）
smobench.config.RANDOM_SEED = 0           # 全局随机种子（默认 42）
smobench.config.CLUSTERING_METHODS = ["leiden", "louvain", "kmeans", "mclust"]
```

---

## 12. 常见问题

### Q: `singularity: command not found`

在 HPC 上需要先加载模块：

```bash
module load singularity
# 或
module load apptainer
```

也可以通过环境变量指定路径：

```bash
export SMOBENCH_SINGULARITY_PATH=/app/apps/singularity/4.2.2/bin/singularity
```

### Q: `CUDA out of memory`

```python
# 使用 CPU
results = smobench.benchmark(..., device="cpu")

# 或指定其他 GPU
results = smobench.benchmark(..., device="cuda:1")
```

### Q: 某个方法报 ImportError

```bash
# 查看该方法需要的依赖
smobench list methods

# 安装对应 extras
pip install smobench[cosmos]
```

如果使用 singularity 隔离，确保对应的 `singularity/envs/{method}/` 前缀已安装。

### Q: `.X.A` AttributeError

某些方法的原始代码使用了 scipy 旧版 API `.A`（已在新版 scipy 中移除）。SMObench 已在 vendor 代码中修复为 `.toarray()`。如果遇到此错误，请更新到最新版本。

### Q: MultiGATE 运行很慢

MultiGATE 使用基于半径的空间图。如果坐标是 Visium 网格格式 [0, ~127]，`rad_cutoff=40` 会导致每个 spot 连接到几乎所有其他 spot。这是原文设定，属于预期行为。

### Q: 如何只运行部分数据集 / 部分方法？

```python
# 指定数据集和方法
results = smobench.benchmark(
    dataset=["Human_Lymph_Nodes", "Human_Tonsils"],
    methods=["SpatialGlue", "SMOPCA"],
)
```

```bash
smobench run --dataset Human_Lymph_Nodes,Human_Tonsils --methods SpatialGlue,SMOPCA
```

### Q: 如何恢复中断的 benchmark？

设置 `save_dir` 后，已完成的方法结果会保存在 h5ad 中。重新运行时，如果检测到已有结果文件，可以通过 `smobench.io.list_methods_in_file()` 检查哪些方法已完成，跳过已完成的。

### Q: 如何并行运行？

```python
results = smobench.benchmark(..., n_jobs=4)  # 4 个并行进程
```

```bash
smobench run --n-jobs 4
```

注意：GPU 方法并行时需要多张 GPU 或足够的显存。
