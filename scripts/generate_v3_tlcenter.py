#!/usr/bin/env python3
"""使用目标会话协方差中心对齐生成 v3 提交候选。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyriemann.geometry.mean import mean_covariance
from pyriemann.geometry.tangentspace import tangent_space
from pyriemann.transfer import TLCenter, encode_domains
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.train_final_tangent import align_scaled, expected_counts
from scripts.train_baseline import sinkhorn_calibrate


TASKS = ("EC", "EO", "Imagery", "SSVEP")
FEATURE_DIR = ROOT / "outputs" / "features"
REPORT_DIR = ROOT / "outputs" / "reports"
SUBMISSION_DIR = ROOT / "outputs" / "submissions"
ROBUST_PREFIXES = ("relative_", "spectral_entropy_", "corr_")


def model() -> LogisticRegression:
    """最终分类器。"""

    return LogisticRegression(C=1.0, max_iter=600, solver="lbfgs", random_state=2026)


def target_alignment(
    source_covariance: np.ndarray,
    target_covariance: np.ndarray,
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """执行目标协方差中心对齐并返回切空间源/目标特征。"""

    all_covariance = np.concatenate([source_covariance, target_covariance], axis=0)
    domain = np.concatenate(
        [
            np.zeros(len(source_covariance), dtype=int),
            np.ones(len(target_covariance), dtype=int),
        ]
    )
    pseudo_labels = np.concatenate(
        [source_labels, np.zeros(len(target_covariance), dtype=int)]
    )
    encoded_covariance, encoded_labels = encode_domains(
        all_covariance,
        pseudo_labels,
        domain,
    )
    aligned = TLCenter(target_domain=1, metric="riemann").fit_transform(
        encoded_covariance,
        encoded_labels,
    )
    reference = mean_covariance(aligned[: len(source_covariance)], metric="riemann")
    tangent = tangent_space(aligned, reference, metric="riemann")
    source_design = np.hstack([tangent[: len(source_covariance)], source_features])
    target_design = np.hstack([tangent[len(source_covariance) :], target_features])
    return source_design, target_design, aligned[: len(source_covariance)], reference


def main() -> None:
    """生成 v3 Excel 提交。"""

    started = time.time()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    train_npz = np.load(FEATURE_DIR / "train_features.npz", allow_pickle=True)
    test_npz = np.load(FEATURE_DIR / "test_features.npz", allow_pickle=True)
    train_x = train_npz["x"]
    test_x = test_npz["x"]
    feature_names = np.asarray(train_npz["names"]).astype(str)
    robust_mask = np.asarray([name.startswith(ROBUST_PREFIXES) for name in feature_names])
    train_meta = pd.read_csv(FEATURE_DIR / "train_metadata.csv")
    test_meta = pd.read_csv(FEATURE_DIR / "test_metadata.csv")
    train_cov = np.load(FEATURE_DIR / "train_covariances.npy")
    test_cov = np.load(FEATURE_DIR / "test_covariances.npy")
    threshold_table = pd.read_csv(REPORT_DIR / "final_tangent_metrics.csv").set_index("task")

    test_meta["inferred_kind"] = np.where(
        test_meta["session_id"].eq("session1"),
        "unknown",
        np.where(test_meta["file_size_bytes"] > 170000, "adversarial", "normal"),
    )
    predicted_label = np.zeros(len(test_meta), dtype=int)
    confidence = np.zeros(len(test_meta), dtype=np.float32)
    accepted_probability = np.zeros((len(test_meta), 20), dtype=np.float32)

    for task in TASKS:
        train_mask = train_meta["task_label"].eq(task).to_numpy()
        train_labels = train_meta.loc[train_mask, "subject_id"].astype(int).to_numpy()
        train_robust = train_x[train_mask][:, robust_mask]
        train_task_cov = train_cov[train_mask]

        test_mask = test_meta["task_label"].eq(task).to_numpy()
        test_indices = np.where(test_mask)[0]
        test_kind = test_meta.loc[test_mask, "inferred_kind"].to_numpy()
        test_robust = test_x[test_mask][:, robust_mask]
        test_task_cov = test_cov[test_mask]
        task_probability = np.zeros((len(test_indices), 20), dtype=np.float32)

        normal_selection = test_kind == "normal"
        adversarial_selection = test_kind == "adversarial"
        normal_probability = None
        if normal_selection.any():
            source_design, target_design, _, _ = target_alignment(
                train_task_cov,
                test_task_cov[normal_selection],
                train_robust,
                test_robust[normal_selection],
                train_labels,
            )
            scaler = StandardScaler().fit(source_design)
            classifier = model().fit(scaler.transform(source_design), train_labels)
            target_scaled = align_scaled(scaler.transform(target_design))
            normal_probability = classifier.predict_proba(target_scaled)
            normal_target = expected_counts(train_labels, int(normal_selection.sum()))
            normal_probability = sinkhorn_calibrate(normal_probability, normal_target)
            task_probability[normal_selection] = normal_probability.astype(np.float32)

        if adversarial_selection.any():
            source_design, target_design, _, _ = target_alignment(
                train_task_cov,
                test_task_cov[adversarial_selection],
                train_robust,
                test_robust[adversarial_selection],
                train_labels,
            )
            scaler = StandardScaler().fit(source_design)
            classifier = model().fit(scaler.transform(source_design), train_labels)
            target_scaled = align_scaled(scaler.transform(target_design))
            direct_probability = classifier.predict_proba(target_scaled)

            if normal_probability is not None and normal_selection.any():
                robust_scaler = StandardScaler().fit(test_robust[normal_selection])
                normal_robust = robust_scaler.transform(test_robust[normal_selection])
                adversarial_robust = robust_scaler.transform(test_robust[adversarial_selection])
                neighbors = NearestNeighbors(
                    n_neighbors=min(5, len(normal_robust)),
                    metric="cosine",
                ).fit(normal_robust)
                distances, indices = neighbors.kneighbors(adversarial_robust)
                weights = 1.0 / np.maximum(distances, 1e-3)
                weights /= weights.sum(axis=1, keepdims=True)
                neighbor_probability = np.einsum(
                    "nk,nkc->nc",
                    weights,
                    normal_probability[indices],
                )
                direct_probability = 0.7 * direct_probability + 0.3 * neighbor_probability

            adversarial_target = expected_counts(
                train_labels,
                int(adversarial_selection.sum()),
            )
            direct_probability = sinkhorn_calibrate(direct_probability, adversarial_target)
            task_probability[adversarial_selection] = direct_probability.astype(np.float32)

        task_confidence = task_probability.max(axis=1)
        labels = np.arange(1, 21)[task_probability.argmax(axis=1)]
        threshold = float(threshold_table.loc[task, "best_threshold"])
        labels = np.where(
            (test_kind == "adversarial") & (task_confidence < threshold),
            0,
            labels,
        )
        labels = np.where(test_kind == "unknown", 0, labels)
        predicted_label[test_indices] = labels
        confidence[test_indices] = task_confidence
        accepted_probability[test_indices] = task_probability

    prediction = test_meta[
        ["sample_id", "task_label", "session_id", "file_size_bytes", "inferred_kind"]
    ].copy()
    prediction["predicted_label"] = predicted_label
    prediction["confidence"] = confidence
    prediction.to_csv(
        REPORT_DIR / "final_predictions_v3.csv",
        index=False,
        encoding="utf-8-sig",
    )

    template = pd.read_csv(
        ROOT / "Brainprint_Competition" / "test.csv",
        dtype={"Sample": str},
    )
    mapping = prediction.set_index("sample_id")["predicted_label"]
    template["Label"] = (
        template["Sample"].str.extract(r"(\d+)")[0].astype(int).map(mapping).astype(int)
    )
    template.to_excel(
        SUBMISSION_DIR / "test_submission_v3.xlsx",
        index=False,
        sheet_name="Sheet1",
    )
    template.to_csv(
        SUBMISSION_DIR / "test_submission_v3.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 与 v2 的一致性只用于工程诊断，不代表真实正确率。
    v2 = pd.read_csv(REPORT_DIR / "final_predictions_v2.csv").set_index("sample_id")
    merged = prediction.set_index("sample_id").join(v2[["predicted_label"]], rsuffix="_v2")
    agreement = {
        kind: float((group["predicted_label"] == group["predicted_label_v2"]).mean())
        for kind, group in merged.groupby("inferred_kind")
    }
    summary = {
        "elapsed_seconds": round(time.time() - started, 3),
        "method": "target covariance center alignment + tangent robust logistic regression",
        "label_counts": template["Label"].value_counts().sort_index().astype(int).to_dict(),
        "v2_agreement": agreement,
        "submission_excel": "outputs/submissions/test_submission_v3.xlsx",
    }
    with (REPORT_DIR / "final_tangent_tlcenter_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file_pointer:
        json.dump(summary, file_pointer, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
