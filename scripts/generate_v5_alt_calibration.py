#!/usr/bin/env python3
"""从 v5 概率缓存生成更高软校准强度的备选提交。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from train_baseline import sinkhorn_calibrate

try:
    from generate_v5_deterministic import PAIRING_PATH, TASKS
except ModuleNotFoundError:
    from generate_v5a_deterministic import PAIRING_PATH, TASKS


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "outputs" / "features"
REPORT_DIR = ROOT / "outputs" / "reports"
SUBMISSION_DIR = ROOT / "outputs" / "submissions"


def save_submission(prediction: pd.DataFrame, name: str) -> pd.DataFrame:
    """写入官方格式 Excel，并返回逐样本模板。"""

    template = pd.read_csv(
        ROOT / "Brainprint_Competition" / "test.csv",
        dtype={"Sample": str},
    )
    mapping = prediction.set_index("sample_id")["predicted_label"]
    template["Label"] = (
        template["Sample"].str.extract(r"(\d+)")[0].astype(int).map(mapping).astype(int)
    )
    template.to_excel(
        SUBMISSION_DIR / f"test_submission_{name}.xlsx",
        index=False,
        sheet_name="Sheet1",
    )
    template.to_csv(
        SUBMISSION_DIR / f"test_submission_{name}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return template


def write_prediction(
    metadata: pd.DataFrame,
    pairing: pd.DataFrame,
    task_probability: np.ndarray,
    name: str,
) -> None:
    """把普通样本概率转换为完整提交文件。"""

    normal_prediction = np.arange(1, 21)[task_probability.argmax(axis=1)]
    twin_map = (
        pairing[["adv_sample_id", "clean_sample_id"]]
        .set_index("adv_sample_id")["clean_sample_id"]
        .to_dict()
    )
    normal_by_sample = pd.Series(
        normal_prediction,
        index=metadata["sample_id"],
    )
    predicted_label = normal_prediction.copy()
    adversarial = metadata["session_id"].eq("session2") & metadata[
        "file_size_bytes"
    ].gt(170000)
    predicted_label[adversarial.to_numpy()] = (
        metadata.loc[adversarial, "sample_id"]
        .map(twin_map)
        .map(normal_by_sample)
        .astype(int)
        .to_numpy()
    )
    predicted_label[metadata["session_id"].eq("session1").to_numpy()] = 0

    prediction = metadata[
        ["sample_id", "task_label", "session_id", "file_size_bytes"]
    ].copy()
    prediction["inferred_kind"] = np.where(
        metadata["session_id"].eq("session1"),
        "unknown",
        np.where(metadata["file_size_bytes"].gt(170000), "adversarial", "normal"),
    )
    prediction["predicted_label"] = predicted_label
    prediction["confidence"] = task_probability.max(axis=1)
    prediction.to_csv(
        REPORT_DIR / f"final_predictions_{name}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    submission = save_submission(prediction, name)
    print(name, submission["Label"].value_counts().sort_index().to_dict())


def main() -> None:
    """生成不同校准强度和共享先验的备选预测。"""

    metadata = pd.read_csv(FEATURE_DIR / "test_metadata.csv")
    pairing = pd.read_csv(PAIRING_PATH)
    train_probability = np.zeros((len(metadata), 20), dtype=np.float32)
    uniform_probability = np.zeros((len(metadata), 20), dtype=np.float32)
    for task in TASKS:
        cached = np.load(REPORT_DIR / f"v5c_probs_{task}.npz")
        sample_index = cached["sample_ids"] - 1
        train_probability[sample_index] = cached["sinkhorn"]
        uniform_probability[sample_index] = sinkhorn_calibrate(
            cached["raw"].astype(np.float64),
            np.full(20, len(sample_index) / 20.0),
        )

    for alpha in (0.50, 0.70):
        task_probability = np.zeros_like(train_probability)
        for task in TASKS:
            cached = np.load(REPORT_DIR / f"v5c_probs_{task}.npz")
            raw = cached["raw"]
            calibrated = cached["sinkhorn"]
            task_probability[cached["sample_ids"] - 1] = (
                (1.0 - alpha) * raw + alpha * calibrated
            )
        name = f"v5_alt_soft{int(alpha * 100):02d}"
        write_prediction(metadata, pairing, task_probability, name)

    write_prediction(metadata, pairing, uniform_probability, "v5_alt_uniform_full")


if __name__ == "__main__":
    main()
