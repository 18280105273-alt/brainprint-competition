#!/usr/bin/env python3
"""生成阶段进度汇报所需图表。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "outputs" / "reports"
FONT_PATH = "/System/Library/Fonts/Hiragino Sans GB.ttc"


def main() -> None:
    """生成数据分布、验证、攻击鲁棒性和预测分布图。"""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    font = FontProperties(fname=FONT_PATH)
    plt.rcParams["axes.unicode_minus"] = False
    train_meta = pd.read_csv(ROOT / "outputs" / "features" / "train_metadata.csv")
    test_meta = pd.read_csv(ROOT / "outputs" / "features" / "test_metadata.csv")
    test_meta["kind"] = np.where(
        test_meta["session_id"].eq("session1"),
        "未知身份候选",
        np.where(test_meta["file_size_bytes"] > 170000, "高频扰动候选", "普通跨会话"),
    )

    tasks = ["EC", "EO", "Imagery", "SSVEP"]
    labels = ["闭眼", "睁眼", "图像想象", "SSVEP"]
    train_counts = [int((train_meta["task_label"] == task).sum()) for task in tasks]
    kind_counts = pd.crosstab(test_meta["task_label"], test_meta["kind"]).reindex(tasks).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.45]})
    axes[0].bar(labels, train_counts, color="#4F81BD")
    axes[0].set_title("训练集各任务样本数", fontproperties=font, fontsize=13)
    axes[0].set_ylabel("文件数", fontproperties=font)
    axes[0].tick_params(axis="x", labelrotation=15)
    for tick in axes[0].get_xticklabels():
        tick.set_fontproperties(font)
    for index, value in enumerate(train_counts):
        axes[0].text(index, value + 70, str(value), ha="center", fontproperties=font, fontsize=9)

    bottom = np.zeros(len(tasks))
    colors = {"普通跨会话": "#4F81BD", "高频扰动候选": "#C0504D", "未知身份候选": "#9BBB59"}
    for kind in ["普通跨会话", "高频扰动候选", "未知身份候选"]:
        values = kind_counts[kind].to_numpy()
        axes[1].bar(labels, values, bottom=bottom, label=kind, color=colors[kind])
        bottom += values
    axes[1].set_title("测试集推断构成", fontproperties=font, fontsize=13)
    axes[1].set_ylabel("文件数", fontproperties=font)
    axes[1].legend(prop=font, frameon=False)
    axes[1].tick_params(axis="x", labelrotation=15)
    for tick in axes[1].get_xticklabels():
        tick.set_fontproperties(font)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "data_distribution.png", dpi=220)
    plt.close(fig)

    metrics = pd.read_csv(REPORT_DIR / "final_tangent_metrics.csv").set_index("task")
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    values = [metrics.loc[task, "clean_accuracy"] * 100 for task in tasks]
    bars = ax.bar(labels, values, color=["#4F81BD", "#9BBB59", "#8064A2", "#F79646"])
    ax.set_ylim(70, 100)
    ax.set_ylabel("时间顺序留出准确率（%）", fontproperties=font)
    ax.set_title("单会话时间顺序验证：先验校准后的组级准确率", fontproperties=font, fontsize=13)
    ax.tick_params(axis="x", labelrotation=15)
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(font)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.6, f"{value:.1f}%", ha="center", fontproperties=font, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "temporal_validation.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(tasks))
    width = 0.34
    for offset, column, label, color in (
        (-width / 2, "clean_accuracy", "干净验证 C", "#4F81BD"),
        (width / 2, "highfreq_A", "高频扰动 A", "#C0504D"),
    ):
        values = [metrics.loc[task, column] * 100 for task in tasks]
        ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=color,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontproperties=font)
    ax.set_ylim(0, 105)
    ax.set_ylim(70, 101)
    ax.set_ylabel("验证准确率或对抗综合分（%）", fontproperties=font)
    ax.set_title("干净识别与高频扰动综合表现", fontproperties=font, fontsize=13)
    ax.legend(prop=font, frameon=False, ncol=2, loc="upper center")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "attack_robustness.png", dpi=220)
    plt.close(fig)

    prediction = pd.read_csv(REPORT_DIR / "final_predictions_v3.csv")
    counts = prediction["predicted_label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(counts.index.astype(str), counts.values, color=["#9BBB59"] + ["#4F81BD"] * 20)
    ax.set_xlabel("预测标签", fontproperties=font)
    ax.set_ylabel("样本数", fontproperties=font)
    ax.set_title("v3 测试预测的标签分布", fontproperties=font, fontsize=13)
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(font)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "prediction_distribution.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
