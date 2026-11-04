# Brainprint Competition

脑纹识别竞赛的可复现实验工程，包含数据审计、频域/空间特征、协方差切空间模型、目标会话对齐、对抗鲁棒性验证和 Excel 提交生成。

## 当前成果

- 训练集：20 名受试者，11,304 个 trial，仅 session1。
- 测试集：22,251 个 trial，官方要求 Excel 提交。
- 数据形状：18 通道、300 Hz、1200 个时间点，每条 4 秒。
- 官方预处理：50 Hz 陷波、1–40 Hz 四阶双向零相位滤波、球面插值、异常片段剔除、逐通道去均值。
- 第一次提交 `v4` 的官方成绩：`54.09`。
- 旧版内部五折约 `99%` 只代表同一 session 内识别，不能用于估计跨会话成绩。
- 第二次提交候选：`v5h_feature_aligned`，使用任务级切空间集成、`TLCenter`、目标手工特征块一次性标准化和跨任务共享均匀先验完整校准。
- 回滚候选：`v5g_target_aligned`、`v5_alt_uniform_full` 和 `v5c_soft03`。
- 7,900 条对抗样本通过原始信号相似度大于 `0.999` 的干净孪生继承标签。
- 测试 session1 的 3,033 条样本统一输出未知身份 `0`。
- 最终 Excel 已通过 22,251 行、顺序、范围、未知拒识和对抗孪生一致性检查。

官方评分：

```text
Score = 100 × C^0.40 × (0.7×RA + 0.3×ASafeR)^0.35 × U^0.25
```

第二次提交文件与复核记录：

- [`outputs/submissions/test_submission_v5_final.xlsx`](outputs/submissions/test_submission_v5_final.xlsx)
- [`outputs/reports/v5_final_validation.json`](outputs/reports/v5_final_validation.json)
- [`outputs/reports/v5a_summary.json`](outputs/reports/v5a_summary.json)
- [`outputs/reports/v5c_calibration_sweep.csv`](outputs/reports/v5c_calibration_sweep.csv)
- [`docs/第二次提交优化记录_2026-11-04.md`](docs/第二次提交优化记录_2026-11-04.md)
- [`docs/独立复核报告_2026-11-01.md`](docs/独立复核报告_2026-11-01.md)

## 项目结构

```text
.
├── README.md
├── requirements.txt
├── run.py
├── src
│   ├── __init__.py
│   ├── __main__.py
│   ├── catalog.py
│   ├── cli.py
│   └── brainprint_comp
│       ├── data.py
│       ├── features.py
│       ├── modeling.py
│       └── scoring.py
├── scripts
│   ├── audit_competition.py
│   ├── extract_features.py
│   ├── extract_covariances.py
│   ├── train_baseline.py
│   ├── train_final_tangent.py
│   ├── generate_v3_tlcenter.py
│   ├── search_ensemble.py
│   ├── generate_v4_ensemble.py
│   ├── generate_v5_deterministic.py
│   ├── generate_v5_alt_calibration.py
│   ├── generate_v5_target_aligned.py
│   ├── make_progress_figures.py
│   └── make_progress_report.py
├── docs
├── outputs
│   ├── figures
│   ├── reports
│   └── submissions
└── data
    └── README.md
```

## 安装

建议 Python 3.10 及以上：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据放置

原始比赛数据不进入 Git。将官方数据放在：

```text
Brainprint_Competition/
├── Train/
├── Test/
└── test.csv
```

## 一键命令

```bash
python -m src --help
python -m src audit
python -m src features
python -m src covariances
python -m src train
python -m src ensemble
python -m src predict
```

也可以直接运行完整流水线：

```bash
python -m src pipeline
```

也可以直接运行脚本：

```bash
.venv/bin/python scripts/audit_competition.py --workers 8
.venv/bin/python scripts/extract_features.py --workers 8
.venv/bin/python scripts/extract_covariances.py --workers 8
.venv/bin/python scripts/train_final_tangent.py
.venv/bin/python scripts/search_ensemble.py
.venv/bin/python scripts/generate_v3_tlcenter.py
.venv/bin/python scripts/generate_v4_ensemble.py
.venv/bin/python scripts/generate_v5_deterministic.py
```

## 结果文件

- `outputs/reports/adv_clean_pairs.csv`：对抗样本与干净孪生的精确配对表。
- `outputs/reports/v5a_summary.json`：各候选的类别分布和代理指标。
- `outputs/reports/v5c_calibration_sweep.csv`：先验校准强度扫描结果。
- `outputs/reports/v5_labeled_stress_summary.json`：通道混合与有标签压力测试。
- `outputs/reports/v5_final_validation.json`：第二次提交的完整结构复核。
- `outputs/reports/v5g_target_aligned_validation.json`：目标标准化候选的独立复核。
- `outputs/reports/v5h_feature_aligned_validation.json`：目标特征块标准化的独立复核。
- `outputs/submissions/test_submission_v5_final.xlsx`：第二次提交主文件。
- `outputs/submissions/test_submission_v5h_feature_aligned.xlsx`：目标特征标准化主候选备份。
- `outputs/submissions/test_submission_v5_alt_soft50.xlsx`：第三次候选，50% 先验校准。
- `outputs/submissions/test_submission_v5_alt_soft70.xlsx`：第三次候选，70% 先验校准。
- `outputs/submissions/test_submission_v5_alt_full.xlsx`：第三次候选，完整训练先验校准。
- `outputs/submissions/test_submission_v5_alt_uniform_full.xlsx`：第三次候选，完整共享均匀先验校准。

## 重要说明

1. 训练集只有 session1，因此训练内部的五折验证不是实际跨会话成绩。
2. session1 全员未知是根据数据会话结构、样本数量和未知身份规则做出的强推断。
3. 7,900 个固定大小样本已通过原始信号相关性与干净样本精确配对。
4. 任何内部代理指标都不能替代组委会官方成绩，第二次提交的官方反馈是下一次决策依据。
