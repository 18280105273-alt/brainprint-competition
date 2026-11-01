# Brainprint Competition

脑纹识别竞赛的可复现实验工程，包含数据审计、频域/空间特征、协方差切空间模型、目标会话对齐、对抗鲁棒性验证和 Excel 提交生成。

## 当前成果

- 训练集：20 名受试者，11,304 个 trial，仅 session1。
- 测试集：22,251 个 trial，官方要求 Excel 提交。
- 数据形状：18 通道、300 Hz、1200 个时间点，每条 4 秒。
- 官方预处理：50 Hz 陷波、1–40 Hz 四阶双向零相位滤波、球面插值、异常片段剔除、逐通道去均值。
- 最终模型：任务级协方差切空间集成 + 目标会话对齐。
- 五折时间分段内部验证：整体 `C=0.9896`。
- 5 dB 高频扰动模拟：整体 `RA=0.9860`、`ASafeR=0.9918`、`A=0.9877`。
- session1 未知身份统一拒识，因此内部按 `U=1` 做评分投影。
- 内部投影分：`99.15`，仅用于模型选择，不是官方成绩。

官方评分：

```text
Score = 100 × C^0.40 × (0.7×RA + 0.3×ASafeR)^0.35 × U^0.25
```

详细阶段报告：

- [`docs/阶段进度汇报_2026-11-01.docx`](docs/阶段进度汇报_2026-11-01.docx)
- [`docs/独立复核报告_2026-11-01.md`](docs/独立复核报告_2026-11-01.md)
- [`outputs/reports/final_tangent_metrics.csv`](outputs/reports/final_tangent_metrics.csv)
- [`outputs/submissions/test_submission_v4.xlsx`](outputs/submissions/test_submission_v4.xlsx)
- [`docs/最终方案说明.md`](docs/最终方案说明.md)

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
```

## 结果文件

- `outputs/reports/final_tangent_metrics.csv`：逐任务 C、RA、ASafeR、A、阈值和内部投影。
- `outputs/reports/final_predictions_v3.csv`：测试逐样本预测和置信度。
- `outputs/submissions/test_submission_v4.xlsx`：最终官方格式 Excel 候选。
- `outputs/reports/final_v4_metrics.csv`：最终逐任务指标。
- `docs/阶段进度汇报_2026-11-01.docx`：图文阶段报告。

## 重要说明

1. 训练集只有 session1，因此训练内部的五折验证不是实际跨会话成绩。
2. session1 全员未知是根据数据会话结构、样本数量和未知身份规则做出的强推断。
3. 7,900 个固定大小样本具有明显额外高频能量，推断为对抗扰动候选。
4. 当前内部投影分不能替代组委会官方成绩。
