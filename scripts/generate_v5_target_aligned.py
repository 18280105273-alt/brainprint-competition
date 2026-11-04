#!/usr/bin/env python3
"""生成任务级目标特征标准化候选，并使用共享均匀先验校准。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from generate_v5_deterministic import (
        PAIRING_PATH,
        TASKS,
        WEIGHTS,
        align_tangent,
    )
except ModuleNotFoundError:
    from generate_v5a_deterministic import (
        PAIRING_PATH,
        TASKS,
        WEIGHTS,
        align_tangent,
    )

from train_baseline import sinkhorn_calibrate


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "outputs" / "features"
REPORT_DIR = ROOT / "outputs" / "reports"
SUBMISSION_DIR = ROOT / "outputs" / "submissions"


def classifier() -> LogisticRegression:
    """创建与主方案一致的逻辑回归分类器。"""

    return LogisticRegression(C=1.0, max_iter=600, solver="lbfgs", random_state=2026)


def aligned_probability(
    task: str,
    train_features: np.ndarray,
    train_covariance: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_covariance: np.ndarray,
    feature_names: np.ndarray,
    robust_mask: np.ndarray,
    relcorr_mask: np.ndarray,
) -> np.ndarray:
    """先做切空间对齐，再只把完整目标集的手工特征块标准化。"""

    source_tangent, target_tangent = align_tangent(
        train_covariance,
        test_covariance,
        train_labels,
    )
    designs = {
        "tan_rob": (train_features[:, robust_mask], test_features[:, robust_mask]),
        "tan_all": (train_features, test_features),
        "tan_relcorr": (train_features[:, relcorr_mask], test_features[:, relcorr_mask]),
    }
    probability = np.zeros((len(test_features), 20), dtype=np.float64)
    for design_name, (source_feature, target_feature) in designs.items():
        source_design = np.hstack([source_tangent, source_feature])
        target_design = np.hstack([target_tangent, target_feature])
        scaler = StandardScaler().fit(source_design)
        source_scaled = scaler.transform(source_design)
        target_scaled = scaler.transform(target_design)
        tangent_dimensions = source_tangent.shape[1]
        target_scaled[:, tangent_dimensions:] = (
            target_scaled[:, tangent_dimensions:]
            - target_scaled[:, tangent_dimensions:].mean(axis=0, keepdims=True)
        ) / np.maximum(
            target_scaled[:, tangent_dimensions:].std(axis=0, keepdims=True),
            1e-8,
        )
        model = classifier().fit(source_scaled, train_labels)
        probability += WEIGHTS[task].get(design_name, 0.0) * model.predict_proba(
            target_scaled
        )
    return probability / sum(WEIGHTS[task].values())


def save_submission(prediction: pd.DataFrame, name: str) -> pd.DataFrame:
    """写入官方格式 Excel 和 CSV。"""

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


def main() -> None:
    """生成并保存目标标准化候选。"""

    train_npz = np.load(FEATURE_DIR / "train_features.npz", allow_pickle=True)
    test_npz = np.load(FEATURE_DIR / "test_features.npz", allow_pickle=True)
    train_features = train_npz["x"]
    test_features = test_npz["x"]
    feature_names = np.asarray(train_npz["names"]).astype(str)
    train_meta = pd.read_csv(FEATURE_DIR / "train_metadata.csv")
    test_meta = pd.read_csv(FEATURE_DIR / "test_metadata.csv")
    train_covariance = np.load(FEATURE_DIR / "train_covariances.npy")
    test_covariance = np.load(FEATURE_DIR / "test_covariances.npy")
    pairing = pd.read_csv(PAIRING_PATH)

    robust_mask = np.asarray(
        [
            name.startswith(("relative_", "spectral_entropy_", "corr_"))
            for name in feature_names
        ]
    )
    relcorr_mask = np.asarray(
        [name.startswith(("relative_", "corr_")) for name in feature_names]
    )
    probability = np.zeros((len(test_meta), 20), dtype=np.float32)
    for task in TASKS:
        train_mask = train_meta["task_label"].eq(task).to_numpy()
        test_mask = test_meta["task_label"].eq(task).to_numpy()
        normal_mask = (
            test_meta.loc[test_mask, "session_id"].eq("session2")
            & test_meta.loc[test_mask, "file_size_bytes"].le(170000)
        ).to_numpy()
        target_indices = np.where(test_mask)[0][normal_mask]
        labels = train_meta.loc[train_mask, "subject_id"].astype(int).to_numpy()
        task_probability = aligned_probability(
            task,
            train_features[train_mask],
            train_covariance[train_mask],
            labels,
            test_features[test_mask][normal_mask],
            test_covariance[test_mask][normal_mask],
            feature_names,
            robust_mask,
            relcorr_mask,
        )
        task_probability = sinkhorn_calibrate(
            task_probability,
            np.full(20, len(target_indices) / 20.0),
        )
        probability[target_indices] = task_probability.astype(np.float32)
        np.savez_compressed(
            REPORT_DIR / f"v5h_probs_{task}.npz",
            probability=task_probability.astype(np.float32),
            sample_ids=test_meta.loc[target_indices, "sample_id"].to_numpy(),
        )

    normal_prediction = np.arange(1, 21)[probability.argmax(axis=1)]
    twin_map = (
        pairing[["adv_sample_id", "clean_sample_id"]]
        .set_index("adv_sample_id")["clean_sample_id"]
        .to_dict()
    )
    normal_by_sample = pd.Series(normal_prediction, index=test_meta["sample_id"])
    predicted_label = normal_prediction.copy()
    adversarial = test_meta["session_id"].eq("session2") & test_meta[
        "file_size_bytes"
    ].gt(170000)
    predicted_label[adversarial.to_numpy()] = (
        test_meta.loc[adversarial, "sample_id"]
        .map(twin_map)
        .map(normal_by_sample)
        .astype(int)
        .to_numpy()
    )
    predicted_label[test_meta["session_id"].eq("session1").to_numpy()] = 0

    prediction = test_meta[
        ["sample_id", "task_label", "session_id", "file_size_bytes"]
    ].copy()
    prediction["inferred_kind"] = np.where(
        test_meta["session_id"].eq("session1"),
        "unknown",
        np.where(test_meta["file_size_bytes"].gt(170000), "adversarial", "normal"),
    )
    prediction["predicted_label"] = predicted_label
    prediction["confidence"] = probability.max(axis=1)
    prediction.to_csv(
        REPORT_DIR / "final_predictions_v5h_feature_aligned.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_submission(prediction, "v5h_feature_aligned")


if __name__ == "__main__":
    main()
