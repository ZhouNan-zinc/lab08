# 实验八 自制语言模型的知识蒸馏（纯 CPU）

本目录实现字符级 GPT 的知识蒸馏：内置合成「仿古诗句」语料，教师 C=160（0.737 M）→ 学生 C=64（0.147 M，压缩 5.00×），包含教师训练、从零训练、带温度 T 的 KD 训练、T×α 4×4 消融、生成样例，以及 FitNet、自蒸馏、动态量化、教师规模扫描五项拓展。课程报告在 `report/`（本地保留，不进入源码仓库）。

## 环境

- Linux（Ubuntu，内核 7.0.0-28-generic），Python 3.12.3
- PyTorch 2.15.0+cpu，matplotlib 3.11.2
- CPU：Intel Core i7-14650HX，24 线程（程序内 `torch.set_num_threads(8)`）
- 内存 15 GiB；`torch.cuda.is_available() == False`，纯 CPU；不下载预训练权重或外部语料

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 运行

```bash
# 1. 环境自检（30/20 步，写入 results/smoke/quick/）
.venv/bin/python distill_lm.py --mode quick

# 2. 训练教师（200 步，写入 results/formal/teacher.pt）
.venv/bin/python distill_lm.py --mode teacher --retrain

# 3. 从零学生 + 蒸馏学生 + 生成 + 曲线（120 步）
.venv/bin/python distill_lm.py --mode all

# 4. T×α 4×4 消融（每组 80 步）
.venv/bin/python distill_lm.py --mode ablate

# 5. 拓展
.venv/bin/python distill_lm.py --mode feat --feat 0.1
.venv/bin/python distill_lm.py --mode quant
.venv/bin/python distill_lm.py --mode selfdistill
.venv/bin/python distill_lm.py --mode scan
.venv/bin/python distill_lm.py --mode ablate --abl-iters 120
```

所有结果写入 `results/`：`formal/`（教师/从零/蒸馏）、`ablate/`（80 步消融）、`ablate120/`（120 步复测）、`ext_feat/`、`ext_quant/`、`ext_self/`、`ext_scan/`、`best/`、`figures/`、`logs/`；`results/smoke/quick/` 与正式结果隔离。

## 输出

- `results/formal/results_all.json`：表 6 三组模型、生成样例、对比指标
- `results/formal/teacher.pt`、`student_scratch.pt`、`student_kd.pt`
- `results/ablate/results_ablate.json`、`results/ablate120/results_ablate.json`
- `results/figures/curves.png`、`heatmap.png`、`ablation_curves.png`
- 每个运行目录的 `run_meta.json`、`env.json`、`self_test.json`
- `results/summary.json`、`results/summary.md`：由 `analyze_results.py` 从原始 JSON 重算

## 复现实验组与结果

数据：训练集 52,456 字符（3000 合成句 + 16 首固定唐诗），留出集 5,143 字符（300 合成句、种子 999、不含固定唐诗；剔除 3 条与训练集完全相同的句子），词表 322。

| 模型 | 参数量 | 步数 | 耗时/s | 留出集 loss | PPL | 与教师一致率 |
|---|---:|---:|---:|---:|---:|---:|
| 教师 C=160 | 737,280 | 200 | 12.7 | 0.7360 | 2.09 | — |
| 学生·从零 C=64 | 147,456 | 120 | 2.9 | 1.0333 | 2.81 | 0.679 |
| 学生·蒸馏 C=64 | 147,456 | 120 | 6.2 | 0.9477 | 2.58 | 0.714 |

蒸馏相对从零：loss −8.29%，PPL 2.81→2.58，一致率 0.679→0.714。T×α 80 步消融最优 T=2/α=0.3（1.1815）；120 步完整复测最优 T=4/α=0.5（0.9359），故正式配置保留 T=4/α=0.5。补充 seed 1/2 的同条件复测相对下降 4.82%/10.70%（三次 seed 平均 7.94%，样本标准差 2.96 个百分点），方向与主表一致。拓展：FitNet feat=0.1 达 0.9271；自蒸馏 0.9792；动态量化 fp32 658.7 KB→int8 316.8 KB（0.48×），loss 0.9477→0.9483；教师 C=96/160/256 的蒸馏收益为 7.19%/8.29%/1.42%。

核验：`.venv/bin/python analyze_results.py` 共 47 项 PASS；报告结构/字体/图片/数字核验 43 项 PASS。实验指南的参考数字来自另一台机器且留出集口径不同，本 README 只列本次实测值。
