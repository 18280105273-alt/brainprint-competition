#!/usr/bin/env python3
"""生成任务级切空间集成 v4 Excel 提交。"""

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
WEIGHTS = {
    "EC": {"tan_rob": 0.7, "tan_all": 0.2, "tan_relcorr": 0.1},
    "EO": {"tan_all": 1.0},
    "Imagery": {"tan_all": 1.0},
    "SSVEP": {"tan_rob": 0.6, "tan_all": 0.4},
}
BEST_ENSEMBLE = {"EC": "e3", "EO": "m2", "Imagery": "m2", "SSVEP": "e1"}


def classifier() -> LogisticRegression:
    """分类器。"""

    return LogisticRegression(C=1.0, max_iter=600, solver="lbfgs", random_state=2026)


def align_covariance(
    source_covariance: np.ndarray,
    target_covariance: np.ndarray,
    source_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """对齐源和目标协方差并返回切空间。"""

    all_covariance = np.concatenate([source_covariance, target_covariance])
    domains = np.concatenate(
        [
            np.zeros(len(source_covariance), dtype=int),
            np.ones(len(target_covariance), dtype=int),
        ]
    )
    pseudo_labels = np.concatenate(
        [source_labels, np.zeros(len(target_covariance), dtype=int)]
    )
    encoded_x, encoded_y = encode_domains(
        all_covariance,
        pseudo_labels,
        domains,
    )
    aligned = TLCenter(target_domain=1, metric="riemann").fit_transform(
        encoded_x,
        encoded_y,
    )
    reference = mean_covariance(aligned[: len(source_covariance)], metric="riemann")
    tangent = tangent_space(aligned, reference, metric="riemann")
    return tangent[: len(source_covariance)], tangent[len(source_covariance) :]


def main() -> None:
    """生成 v4 Excel 并做结构比较。"""

    started = time.time()
    train_npz = np.load(FEATURE_DIR / "train_features.npz", allow_pickle=True)
    test_npz = np.load(FEATURE_DIR / "test_features.npz", allow_pickle=True)
    train_x = train_npz["x"]
    test_x = test_npz["x"]
    names = np.asarray(train_npz["names"]).astype(str)
    train_meta = pd.read_csv(FEATURE_DIR / "train_metadata.csv")
    test_meta = pd.read_csv(FEATURE_DIR / "test_metadata.csv")
    train_cov = np.load(FEATURE_DIR / "train_covariances.npy")
    test_cov = np.load(FEATURE_DIR / "test_covariances.npy")
    robust_mask = np.asarray(
        [name.startswith(("relative_", "spectral_entropy_", "corr_")) for name in names]
    )
    relcorr_mask = np.asarray(
        [name.startswith(("relative_", "corr_")) for name in names]
    )
    ensemble_search = pd.read_csv(REPORT_DIR / "ensemble_search.csv").set_index(
        ["task", "ensemble"]
    )
    thresholds = {
        task: float(ensemble_search.loc[(task, BEST_ENSEMBLE[task]), "best_threshold"])
        for task in TASKS
    }

    test_meta["inferred_kind"] = np.where(
        test_meta["session_id"].eq("session1"),
        "unknown",
        np.where(test_meta["file_size_bytes"] > 170000, "adversarial", "normal"),
    )
    predicted_label = np.zeros(len(test_meta), dtype=int)
    confidence = np.zeros(len(test_meta), dtype=np.float64)

    for task in TASKS:
        train_mask = train_meta["task_label"].eq(task).to_numpy()
        source_labels = train_meta.loc[train_mask, "subject_id"].astype(int).to_numpy()
        source_features = {
            "tan_rob": train_x[train_mask][:, robust_mask],
            "tan_all": train_x[train_mask],
            "tan_relcorr": train_x[train_mask][:, relcorr_mask],
        }
        source_covariance = train_cov[train_mask]

        test_mask = test_meta["task_label"].eq(task).to_numpy()
        test_indices = np.where(test_mask)[0]
        kinds = test_meta.loc[test_mask, "inferred_kind"].to_numpy()
        test_features = {
            "tan_rob": test_x[test_mask][:, robust_mask],
            "tan_all": test_x[test_mask],
            "tan_relcorr": test_x[test_mask][:, relcorr_mask],
        }
        task_probability = np.zeros((len(test_indices), 20), dtype=np.float32)
        normal_probability = None

        normal_mask = kinds == "normal"
        if normal_mask.any():
            source_tangent, target_tangent = align_covariance(
                source_covariance,
                test_cov[test_mask][normal_mask],
                source_labels,
            )
            probabilities = {}
            for design_name, weight in WEIGHTS[task].items():
                source_design = np.hstack(
                    [source_tangent, source_features[design_name]]
                )
                target_design = np.hstack(
                    [target_tangent, test_features[design_name][normal_mask]]
                )
                scaler = StandardScaler().fit(source_design)
                model = classifier().fit(
                    scaler.transform(source_design),
                    source_labels,
                )
                probabilities[design_name] = model.predict_proba(
                    align_scaled(scaler.transform(target_design))
                )
            normal_probability = sum(
                WEIGHTS[task][name] * probability
                for name, probability in probabilities.items()
            )
            normal_probability /= sum(WEIGHTS[task].values())
            normal_probability = sinkhorn_calibrate(
                normal_probability,
                expected_counts(source_labels, int(normal_mask.sum())),
            )
            task_probability[normal_mask] = normal_probability.astype(np.float32)

        adversarial_mask = kinds == "adversarial"
        if adversarial_mask.any():
            source_tangent, target_tangent = align_covariance(
                source_covariance,
                test_cov[test_mask][adversarial_mask],
                source_labels,
            )
            probabilities = {}
            for design_name in WEIGHTS[task]:
                source_design = np.hstack(
                    [source_tangent, source_features[design_name]]
                )
                target_design = np.hstack(
                    [target_tangent, test_features[design_name][adversarial_mask]]
                )
                scaler = StandardScaler().fit(source_design)
                model = classifier().fit(
                    scaler.transform(source_design),
                    source_labels,
                )
                probabilities[design_name] = model.predict_proba(
                    align_scaled(scaler.transform(target_design))
                )
            adversarial_probability = sum(
                WEIGHTS[task][name] * probability
                for name, probability in probabilities.items()
            )
            adversarial_probability /= sum(WEIGHTS[task].values())

            if normal_probability is not None:
                robust_scaler = StandardScaler().fit(
                    test_features["tan_rob"][normal_mask]
                )
                normal_robust = robust_scaler.transform(
                    test_features["tan_rob"][normal_mask]
                )
                adversarial_robust = robust_scaler.transform(
                    test_features["tan_rob"][adversarial_mask]
                )
                neighbors = NearestNeighbors(
                    n_neighbors=min(5, len(normal_robust)),
                    metric="cosine",
                ).fit(normal_robust)
                distance, indices = neighbors.kneighbors(adversarial_robust)
                weights = 1.0 / np.maximum(distance, 1e-3)
                weights /= weights.sum(axis=1, keepdims=True)
                propagated = np.einsum(
                    "nk,nkc->nc",
                    weights,
                    normal_probability[indices],
                )
                adversarial_probability = (
                    0.7 * adversarial_probability + 0.3 * propagated
                )
            adversarial_probability = sinkhorn_calibrate(
                adversarial_probability,
                expected_counts(source_labels, int(adversarial_mask.sum())),
            )
            task_probability[adversarial_mask] = adversarial_probability.astype(np.float32)

        task_confidence = task_probability.max(axis=1)
        task_labels = np.arange(1, 21)[task_probability.argmax(axis=1)]
        threshold = thresholds[task]
        task_labels = np.where(
            (kinds == "adversarial") & (task_confidence < threshold),
            0,
            task_labels,
        )
        task_labels = np.where(kinds == "unknown", 0, task_labels)
        predicted_label[test_indices] = task_labels
        confidence[test_indices] = task_confidence

    prediction = test_meta[
        ["sample_id", "task_label", "session_id", "file_size_bytes", "inferred_kind"]
    ].copy()
    prediction["predicted_label"] = predicted_label
    prediction["confidence"] = confidence
    prediction.to_csv(
        REPORT_DIR / "final_predictions_v4.csv",
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
        SUBMISSION_DIR / "test_submission_v4.xlsx",
        index=False,
        sheet_name="Sheet1",
    )
    template.to_csv(
        SUBMISSION_DIR / "test_submission_v4.csv",
        index=False,
        encoding="utf-8-sig",
    )

    v3 = pd.read_csv(REPORT_DIR / "final_predictions_v3.csv").set_index("sample_id")
    merged = prediction.set_index("sample_id").join(
        v3[["predicted_label"]],
        rsuffix="_v3",
    )
    agreement = {
        kind: float(
            (group["predicted_label"] == group["predicted_label_v3"]).mean()
        )
        for kind, group in merged.groupby("inferred_kind")
    }
    summary = {
        "elapsed_seconds": round(time.time() - started, 3),
        "method": "task-specific tangent ensemble + target covariance alignment",
        "weights": WEIGHTS,
        "label_counts": template["Label"].value_counts().sort_index().astype(int).to_dict(),
        "v3_agreement": agreement,
        "submission_excel": "outputs/submissions/test_submission_v4.xlsx",
    }
    metric_rows = []
    for task in TASKS:
        row = ensemble_search.loc[(task, BEST_ENSEMBLE[task])]
        clean_accuracy = float(row["clean_accuracy"])
        adversarial_score = float(row["best_A"])
        projected = 0.0
        if clean_accuracy > 0 and adversarial_score > 0:
            projected = 100.0 * clean_accuracy**0.40 * adversarial_score**0.35
        metric_rows.append(
            {
                "task": task,
                "ensemble": BEST_ENSEMBLE[task],
                "clean_accuracy": clean_accuracy,
                "highfreq_accuracy": float(row["highfreq_accuracy"]),
                "white_noise_accuracy": float(row["white_accuracy"]),
                "best_threshold": float(row["best_threshold"]),
                "RA": float(row["best_RA"]),
                "ASafeR": float(row["best_ASafeR"]),
                "A": adversarial_score,
                "projected_score": projected,
            }
        )
    final_metrics = pd.DataFrame(metric_rows)
    final_metrics.to_csv(
        REPORT_DIR / "final_v4_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    normal_counts = [
        int(
            (
                test_meta["task_label"].eq(task)
                & test_meta["inferred_kind"].eq("normal")
            ).sum()
        )
        for task in TASKS
    ]
    adversarial_counts = [
        int(
            (
                test_meta["task_label"].eq(task)
                & test_meta["inferred_kind"].eq("adversarial")
            ).sum()
        )
        for task in TASKS
    ]
    overall_clean = float(np.average(final_metrics["clean_accuracy"], weights=normal_counts))
    overall_ra = float(np.average(final_metrics["RA"], weights=adversarial_counts))
    overall_safe = float(np.average(final_metrics["ASafeR"], weights=adversarial_counts))
    overall_a = 0.7 * overall_ra + 0.3 * overall_safe
    summary["validation"] = {
        "C": overall_clean,
        "RA": overall_ra,
        "ASafeR": overall_safe,
        "A": overall_a,
        "projected_score": 100.0 * overall_clean**0.40 * overall_a**0.35,
    }
    with (REPORT_DIR / "final_v4_summary.json").open("w", encoding="utf-8") as file_pointer:
        json.dump(summary, file_pointer, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
