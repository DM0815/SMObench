# _myx_Scripts 目录结构规范

## 核心原则

1. **所有代码放在 `_myx_` 开头的文件夹内**
2. **不修改原项目代码**（`Eval/`, `Methods/`, `Scripts/`）— 通过 import/wrapper 调用
3. **主文件 vs 调试文件严格分离**：调试代码统一放 `_debug/` 子目录
4. **主文件 = 开源级代码**：干净、有CLI接口、有docstring
5. **PBS脚本**统一放 `pbs/` 子目录

## 目录结构

```
_myx_Scripts/
├── evaluation/                  # 指标计算
│   ├── eval_vertical.py         #   垂直指标 (SC + BioC)
│   ├── eval_horizontal.py       #   水平指标 (SC + BioC + BER)
│   ├── compute_cmgtc.py         #   CM-GTC 计算
│   ├── aggregate_scores.py      #   汇总所有指标 → 最终评分
│   ├── pbs/                     #   PBS 提交脚本
│   └── _debug/                  #   调试 / 测试脚本
├── visualization/               # 可视化（Figure 生成）
│   ├── plot_*.py                #   各 Figure 绘图主文件
│   └── _debug/
├── vertical_integration/        # 垂直整合：per-method 运行脚本
├── horizontal_integration/      # 水平整合：per-method 运行脚本
├── scalability/                 # 可扩展性测试
└── data_preparation/            # 数据准备

_myx_Results/
├── adata/                       # Embedding h5ad 文件
│   ├── vertical_integration/    #   Method/Dataset/xxx.h5ad
│   └── horizontal_integration/
├── evaluation/                  # 指标计算输出
│   ├── vertical/                #   per-method per-dataset CSV
│   ├── horizontal/
│   ├── cmgtc/
│   └── summary/                 #   最终汇总表 (vertical_summary.csv 等)
├── scalability/                 # 可扩展性结果
├── figures/                     # 生成的论文图片
│   ├── fig2/ ~ fig5/
└── _debug/                      # 调试输出
```

## 命名规范

| 类型 | 格式 | 示例 |
|------|------|------|
| 主文件 | `动词_名词.py` | `eval_vertical.py`, `compute_cmgtc.py` |
| PBS脚本 | `pbs_动词_名词.pbs` | `pbs_eval_vertical.pbs` |
| 绘图脚本 | `plot_描述.py` | `plot_radar.py`, `plot_sc_bioc_scatter.py` |
| 调试文件 | 放 `_debug/` 目录 | `_debug/test_single_method.py` |
