# Scalability Analysis - 修改记录

## 概述

为 SMOBench 添加 scalability analysis（计算时间 + 内存 vs 细胞数），参照论文 [3] (Hu et al., Nature Methods 2024) 方法。

- **RNA_ADT**: Mouse_Thymus (17,824 cells) → [1000, 2500, 5000, 10000, 17824], 5 repeats
- **RNA_ATAC**: Mouse_Brain (37,885 cells) → [1000, 2500, 5000, 10000, 20000, 37885], 5 repeats
- **13 个方法**, 测量 wall time + GPU memory + system RSS
- **总计**: 24 个 PBS job, 655 个独立 run

---

## 新增脚本 (_myx_Scripts/scalability/)

| 文件 | 功能 |
|------|------|
| `prepare_scalability_data.py` | 从 fusion 数据生成子采样 h5ad |
| `run_scalability.py` | 核心编排器：调用方法脚本 + 监控内存 + 保存 JSON |
| `submit_scalability.py` | 生成/提交 per-method PBS 脚本 |
| `collect_scalability_results.py` | 收集 JSON 结果 + 生成时间/内存图表 |

---

## Conda 环境映射

| 环境 | 方法 |
|------|------|
| `_Proj1_1` | CANDIES, COSMOS, MISO, PRAGA, SpatialGlue, SpaMultiVAE, SMOPCA, SpaBalance, SpaFusion |
| `_Proj1_1_PRESENT` | PRESENT, SpaMI |
| `_Proj1_1_SpaMosaic` | SpaMosaic |
| `_Proj1_1_SpaMV` | SpaMV |

---

## 方法脚本修改详情

### 1. SpaMosaic (`horizontal_integration/SpaMosaic/run_spamosaic.py`)

**问题 1: GLIBC_2.32 错误**
- `_Proj1_1` 环境的 torch_scatter 需要 GLIBC_2.32（系统只有 2.28）
- **修复**: 切换到 `_Proj1_1_SpaMosaic` 环境（torch-scatter 2.1.2+pt21cu121，只需 GLIBC_2.14）

**问题 2: numba float64 错误**
- 子采样数据为 float64，numba 在 scanpy 中要求 float32
- **修复**: 数据加载后添加 `.astype(np.float32)` 转换
```python
if hasattr(adata_rna_full.X, 'dtype') and adata_rna_full.X.dtype != np.float32:
    adata_rna_full.X = adata_rna_full.X.astype(np.float32)
```

**问题 3: loess singularity (seurat_v3 HVG)**
- 小样本数据中部分基因在所有细胞中为 0，导致 loess 拟合失败
- **修复**: RNA_preprocess 前添加预过滤
```python
for ad in input_dict['rna']:
    sc.pp.filter_genes(ad, min_cells=1)
```

**问题 4: harmonypy 版本不兼容**
- `harmonypy 0.0.10` 的 `run_harmony()` 不支持 `batch_key` 参数
- Methods/SpaMosaic 代码调用 `harmonize(batch_key="batch", use_gpu=...)`
- **修复**: 安装 `harmony-pytorch`（提供兼容的 `harmonize()` 函数）
```bash
conda run -n _Proj1_1_SpaMosaic python -m pip install --user harmony-pytorch
```

---

### 2. SpaMultiVAE (`horizontal_integration/SpaMultiVAE/run_spamultivae.py`)

**问题 1: 构造函数 API 完全错误**
- 原代码传 `gene_exp=, protein_exp=` 等参数，实际 API 为 `gene_dim=, protein_dim=`
- **修复**: 完全重写，参照官方 `run_spaMultiVAE.py` 示例：
```python
model = SPAMULTIVAE(gene_dim=adata1.n_vars, protein_dim=adata2.n_vars, ...)
model.train_model(pos=loc, gene_ncounts=adata1.X, gene_raw_counts=adata1.raw.X, ...)
embeddings = model.batching_latent_samples(X=loc, gene_Y=adata1.X, protein_Y=adata2.X, ...)
```

**问题 2: normalize() 参数错误**
- `normalize()` 不接受 `copy`, `target_sum`, `log1p` 参数
- **修复**: 改为 `normalize(adata, size_factors=True, normalize_input=True, logtrans_input=True)`

**问题 3: seurat_v3 HVG 报错**
- `Extrapolation not allowed with blending`
- **修复**: HVG flavor 从 `seurat_v3` 改为 `cell_ranger`

**问题 4: normalize 内部 filter_cells 导致维度不对齐**
- `normalize()` 内部调用 `sc.pp.filter_cells()` 删除部分细胞
- 导致 `adata1.X` 行数与 `loc`（坐标）不一致，adata1 和 adata2 行数也可能不同
- **修复**: 传 `filter_min_counts=False`（我们已预过滤零计数基因/蛋白质）
```python
adata1 = normalize(adata1, filter_min_counts=False, size_factors=True, ...)
```

**问题 5: AnnData 不拷贝数据导致原地修改**
- `sc.AnnData(protein_raw)` 不做拷贝
- 第一次 `normalize()` 的 `sc.pp.log1p()` 原地修改了 `protein_raw`
- 第二次创建 `adata2_no_scale` 时使用已被 log1p 的数据，再次 log1p 导致错误
- **修复**: 创建 AnnData 时显式拷贝
```python
adata1 = sc.AnnData(gene_raw.copy(), dtype="float64")
adata2 = sc.AnnData(protein_raw.copy(), dtype="float64")
adata2_no_scale = sc.AnnData(protein_raw.copy(), dtype="float64")
```

**问题 6: 原始数据清洗**
- 添加 NaN/Inf 替换和非负约束（防止 log1p 产生 NaN）
```python
gene_raw = np.nan_to_num(gene_raw, nan=0.0, posinf=0.0, neginf=0.0)
gene_raw = np.maximum(gene_raw, 0)
```

---

### 3. PRESENT (`horizontal_integration/PRESENT/run_present.py`)

**问题 1: import 路径错误**
- `Methods/PRESENT/` 没有 `__init__.py`，实际包在 `Methods/PRESENT/PRESENT/`
- **修复**: 添加正确的 sys.path
```python
sys.path.insert(0, os.path.join(project_root, "Methods", "PRESENT"))
```

**问题 2: 函数名错误**
- `PRESENT_BC_function` 不存在
- **修复**: 改为 `PRESENT_function`

**问题 3: ADT 模式缺少参数**
- ADT 模式下 `present_args['adata_adt'] = adata_adt` 缺失（原为 `pass`）
- **修复**: 添加 ADT 数据传递

---

### 4. SpaFusion (`horizontal_integration/SpaFusion/run_SpaFusion.py`)

**问题 1: import 路径错误**
- `clustering`, `distribution_loss`, `target_distribution`, `assignment` 在 `utils.py` 而非 `evaluate.py`
- **修复**: `from utils import clustering, distribution_loss, target_distribution, assignment`

**问题 2: dtype 不匹配**
- AMP autocast 产生 float32 Z，但 KMeans 返回 float64 centers
- `model.cluster_centers1.data` 赋值时类型冲突
- **修复**: 显式指定 dtype
```python
model.cluster_centers1.data = torch.tensor(centers1, dtype=torch.float32).to(device)
```

---

### 5. SpaMI (`horizontal_integration/SpaMI/run_SpaMI.py`)

**问题 1: 环境不兼容**
- `_Proj1_1` 环境的 torch_geometric GLIBC 问题，导致退回 CPU 运行极慢（超过 7200s timeout）
- **修复**: 切换到 `_Proj1_1_PRESENT` 环境（兼容的 torch-scatter）

**问题 2: scalability 数据集参数查找失败**
- `train()` 根据 dataset 名查找超参数，`scalability_Mouse_Thymus` 找不到
- **修复**: strip "scalability_" 前缀
```python
dataset_for_train = args.dataset.replace("scalability_", "") if args.dataset.startswith("scalability_") else args.dataset
```

**问题 3: 缺少 POT 包**
- `_Proj1_1_PRESENT` 环境没有 `pot` 包
- **修复**: `python -m pip install --user "POT==0.9.0" --only-binary=:all:`

---

### 6. SpaMV (`horizontal_integration/SpaMV/run_spamv.py`)

**问题: ADT 特征数太少被过滤**
- ADT 只有 10-16 个特征，`min_genes=200` 会过滤掉所有细胞
- **修复**: ADT 模式下设置 `min_genes=1`
```python
min_genes_val = 1 if modality == 'ADT' else 200
```

---

### 7. CANDIES, COSMOS, MISO, PRAGA, SpatialGlue, SMOPCA, SpaBalance

这些方法**无需修改**，直接通过 scalability 测试。

---

## 数据更新

### Mouse_Thymus Fusion ADT 蛋白质数量变更
- **旧**: 10 个蛋白质（未经名称规范化匹配）
- **新**: 16 个蛋白质（经名称规范化后匹配更多共有蛋白质）
- 蛋白质列表: cd11b, cd11c, cd169, cd19, cd29, cd3, cd31, cd4, cd44, cd45rb220, cd5, cd68, cd8a, cd902, f480, igg2a
- 旧文件备份为: `Mouse_Thymus_Fusion_ADT.h5ad.bak_10proteins`
- 子采样数据已用新 16 蛋白数据重新生成

---

## 测试结果 (1000 cells, RNA_ADT)

| 方法 | 时间(s) | GPU(MB) | RSS(MB) | 状态 |
|------|---------|---------|---------|------|
| CANDIES | 310 | 1407 | 2475 | OK |
| COSMOS | 200 | 1753 | 1767 | OK |
| MISO | 50 | 639 | 1683 | OK |
| PRAGA | 832 | 813 | 2516 | OK |
| SpaMosaic | 50 | 1063 | 1458 | OK |
| SpatialGlue | 82 | 641 | 1712 | OK |
| SpaMultiVAE | 1027 | 15679 | 2435 | OK |
| PRESENT | 156 | 5 | 1429 | OK |
| SMOPCA | 172 | 5 | 1150 | OK |
| SpaBalance | 160 | 695 | 1919 | OK |
| SpaMI | 94 | 685 | 1504 | OK |
| SpaMV | 280 | 801 | 2123 | OK |
| SpaFusion | 213 | 905 | 2160 | OK |

---

## 全量运行修复 - 第二批 (2026-03-14)

### 7. SpaMV - 缺少 scikit-misc

**文件**: 环境 `_Proj1_1_SpaMV`
**错误**: `ImportError: Please install skmisc package via pip install --user scikit-misc`
**原因**: scanpy 的 `highly_variable_genes(flavor='seurat_v3')` 需要 scikit-misc
**修复**: `conda run -n _Proj1_1_SpaMV python -m pip install --user scikit-misc`

### 8. SpaMosaic ATAC - 缺少空间坐标

**文件**: `_myx_Scripts/horizontal_integration/SpaMosaic/run_spamosaic.py`
**错误**: `KeyError: 'spatial'` — SpaMosaic 构建空间图时需要 `obsm['spatial']`
**原因**: Fusion ATAC 数据没有 `obsm['spatial']`（仅 RNA 有），但两者共享相同细胞
**修复**: 在 fusion_mode 加载后，从 RNA 复制空间坐标到 ATAC

```python
# Copy spatial coordinates from RNA to other modality if missing
if 'spatial' in adata_rna_full.obsm and 'spatial' not in adata_other_full.obsm:
    adata_other_full.obsm['spatial'] = adata_rna_full.obsm['spatial'].copy()
```

### 9. PRESENT ATAC - 非整数 ATAC 值导致 Poisson 分布报错

**文件**: `_myx_Scripts/horizontal_integration/PRESENT/run_present.py`
**错误**: `ValueError: Expected value argument to be within the support (IntegerGreaterThan(lower_bound=0))`
**原因**: Fusion ATAC 数据是 TF-IDF 归一化后的浮点值（如 2.2431），PRESENT 的 Poisson 模型要求整数计数
**修复**: 在 fusion_mode 加载 ATAC 数据后，四舍五入为整数

```python
if modality == 'ATAC':
    if sp.issparse(adata_other_full.X):
        adata_other_full.X.data = np.round(adata_other_full.X.data).clip(0)
        adata_other_full.X.eliminate_zeros()
    else:
        adata_other_full.X = np.round(adata_other_full.X).clip(0)
```

### 重提交任务 (2026-03-14)

| Job ID | 方法 | 模态 | 原因 |
|--------|------|------|------|
| 13172983 | SpaMV | ADT | scikit-misc 修复 |
| 13172984 | SpaMV | ATAC | scikit-misc 修复 |
| 13172985 | SpaMosaic | ATAC | spatial 坐标修复 |
| 13172986 | PRESENT | ATAC | 整数计数修复 |
| 13172987 | SpaFusion | ADT | adj cache 修复(前次排队未运行) |

---

## 全量运行修复 - 第三批 (2026-03-15)

### 10. SpaMV - 缺少 igraph

**文件**: 环境 `_Proj1_1_SpaMV`
**错误**: `ModuleNotFoundError: No module named 'igraph'` — `sc.tl.louvain` 需要 igraph
**修复**: `conda run -n _Proj1_1_SpaMV pip install python-igraph --user`

### 11. Per-run timeout 过短

**文件**: `_myx_Scripts/scalability/run_scalability.py` 第198行
**错误**: 多个方法在大 cell count 时 `wall_time=7200s` 被 timeout kill
**影响**: CANDIES(5), COSMOS(3), PRAGA(3), SMOPCA(8), SpaBalance(9) = 28 runs
**修复**: `timeout=7200` → `timeout=21600` (2h → 6h)

### 预期极限（OOM，无需修复）

| 方法 | 模态 | 最大成功 cell count | 失败原因 |
|------|------|-------------------|---------|
| MISO | ATAC | 20000 | CUDA OOM @ 37885 |
| SpaMultiVAE | ADT | 1000 | GPU OOM (GP kernel O(n²)) |
| SpaFusion | ADT | 10000 | CUDA OOM @ 17824 (39GB) |
| SpatialGlue | ADT | 1000 rep2 | 偶发失败 |

### 重提交任务 (2026-03-15)

| Job ID | 内容 | 原因 |
|--------|------|------|
| 13178336 | SpaMV ADT (25 runs) | igraph 修复 |
| 13178337 | SpaMV ATAC (30 runs) | igraph 修复 |
| 13178343 | 5方法 timeout 重跑 (28 runs) | timeout 6h 修复 |
