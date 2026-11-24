# 实验运行与复现说明

本代码包含 Section 5.3 和 EC.7 的数值实验。[EXPERIMENTS.md](EXPERIMENTS.md)
按指定 PDF 的 Figure 1、Figures EC.2–EC.6 和 Tables EC.3–EC.7，
列出了页码、代码入口和输出文件。本指南说明如何使用随包提供的输入数据重算结果、
画图和导出表格。

## 1. 安装和一键运行

在代码根目录执行：

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-replay.txt
python scripts/reproduce_paper.py --workers 4 --output ../paper-results
```

输出目录必须尚不存在。程序保留原输入和存档结果，在新目录中保存重算结果。
从 GitHub 克隆时，五份大输入以压缩文件保存在 `data/inputs/`；上述一键命令会先
自动恢复到实验目录。需要约 1.6 GB 空闲磁盘空间。从 ZIP 解压时输入文件已直接包含。
`--workers` 控制并行进程数。细网格 DP 的最大主数组约 1.54 GB，另需工作缓冲区，
建议预留数 GB 内存和磁盘空间。

一键流程包含九个步骤，日志为 `01-*.log` 至 `09-*.log`，总体状态为 `RUN.json`。
失败时程序停止并指出对应日志。数值核验的绝对误差阈值为 `1e-7`。

## 2. 输出在哪里

| 内容 | 输出目录中的位置 |
|---|---|
| Figure 1：Section 5.3 的两幅示意图 | `figure1/illustration/figure_{regret,ratio}.png` |
| Tables EC.3–EC.4：实例及闭式解、LP、DP 比较 | `verification/verification/reproduced_results/comparison.md`、同目录 `results.json` |
| Tables EC.5–EC.6：常数成本策略表的重算值 | `tables/result_ratio*.json`、同目录 `result_rgt*.json` |
| Figures EC.3–EC.4：shift 和 demand/horizon 的策略值 | `predictions/result_{ratio,rgt}_{shift,demand_horizon}.json` |
| Figures EC.5–EC.6：变成本实验的策略值 | `changing-cost/recomputed_{ratio,regret}.json` |
| Figures EC.2–EC.4：常数成本实验图 | `figures/constant_cost/results/picture/` |
| Figures EC.5–EC.6：变成本散点图、Pareto 图 | `figures/changing_cost/result/` |
| Tables EC.5–EC.7：表格 CSV | `exported-tables/` |

各数值重算目录中的 `verification.json` 保存检查数量及最大误差。

## 3. 单独运行某一组

以下命令在代码根目录运行，输出目录均应尚不存在。
如果是 GitHub 克隆，先运行 `python scripts/restore_inputs.py`；该步骤会校验恢复文件的 SHA-256。

```sh
# 价格成本比示意图
python scripts/reproduce.py figure1 --output ../figure1-only

# 五个小实例：闭式解、LP、网格 0.0005 的 DP
python scripts/reproduce.py verification --output ../verification-only

# 两张常数成本表，包含网格 1 和 4 的 DP
python scripts/verify_tables.py --dp --vectorized-dp --workers 4 --output ../tables-only

# 偏移预测和需求随时域变化的实验值
python scripts/replay_prediction_figures.py --output ../prediction-values

# 变成本：150 个配置，每个配置 100 个输入样本
python scripts/replay_changing_cost_all.py --workers 4 --output ../changing-values

# 用重算性能值和存档计时生成实验图
python scripts/build_replayed_figures.py --predictions ../prediction-values \
  --changing-cost ../changing-values --output ../experiment-figures

# 导出表格的完整精度统计数据
python scripts/export_tables.py --output ../experiment-tables
```

若只想从已有结果重新画图，不重算算法：

```sh
python scripts/reproduce.py plots --output ../plots-from-results
```

## 4. 输入数据与计算约定

- 输入 JSON 是复现材料的一部分，不应删除。一键复现使用这些固定样本，不重新采样。
- 两张常数成本表使用各 80 个存档样本；shift 有 101 组、每组 80 个样本；
  demand/horizon 有 40 组、每组 100 个样本。
- 变成本重算覆盖三个成本舍入参数下的 150 个配置，每个配置 100 个样本。
  绘图保留全部 150 个成本舍入配置，排除最后 50 个原始成本参考配置。
- 生成器的 horizon 计数包含终端观测，容量向量长度代表实际订货日数。
- 不一致预测下，区间策略先计算端点或中点公式，再把动作截断到容量范围；
  midpoint 是先取原始上下界平均，再截断。
- 部分 ratio 文件沿用 `rgt_up/rgt_lw/rgt_mid` 键名；目标类型以文件名为准。

## 5. 运行时间与版本兼容

运行时间图、表使用原始测量记录。不同机器上的新运行时间不要求与这些秒数一致；
加速核验产生的耗时也不替换历史计时。性能值和运行时间的来源分别记录在输出中。

快速重算使用经过原实现对照检查的 NumPy DP 和 Numba phase-reduction 内核。
原始求解循环仍保留。去掉 `verify_tables.py` 的 `--vectorized-dp` 可运行原列表版
DP，但耗时更长。Numba 编译关闭 fastmath。

Python 3.12 改变了浮点 `sum()` 的实现。为保持存档结果的计算顺序，变成本代码
通过 `changing_cost/numerics.py` 显式从左到右求和。此兼容处理是为了重放历史
数值行为，不是精度改进声明。参考：[Python sum 文档](https://docs.python.org/3.12/library/functions.html#sum)。

## 6. 已执行的检查

[VALIDATION.md](VALIDATION.md) 总结了实际运行范围。
`validation/summary.json` 保存最近一次完整复验的九步状态、数量和误差。
