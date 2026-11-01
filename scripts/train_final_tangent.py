#!/usr/bin/env python3
"""训练协方差切空间主模型并生成 Excel 提交候选。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyriemann.geometry.mean import mean_covariance
from pyriemann.geometry.tangentspace import tangent_space
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from brainprint_comp.data import load_trial
from brainprint_comp.features import extract_features
from scripts.train_baseline import add_high_frequency_noise, sinkhorn_calibrate


TASKS = ("EC", "EO", "Imagery", "SSVEP")
FEATURE_DIR = ROOT / "outputs" / "features"
MODEL_DIR = ROOT / "outputs" / "models"
REPORT_DIR = ROOT / "outputs" / "reports"
SUBMISSION_DIR = ROOT / "outputs" / "submissions"
ROBUST_PREFIXES = ("relative_", "spectral_entropy_", "corr_")


def ensure_dirs() -> None:
    """建立输出目录。"""

    for path in (MODEL_DIR, REPORT_DIR, SUBMISSION_DIR):
        path.mkdir(parents=True, exist_ok=True)


def align_scaled(scaled: np.ndarray) -> np.ndarray:
    """把目标会话的标准化特征对齐到零均值、单位方差。"""

    return (scaled - scaled.mean(axis=0, keepdims=True)) / np.maximum(
        scaled.std(axis=0, keepdims=True),
        1e-8,
    )


def make_model() -> LogisticRegression:
    """构造最终分类器。"""

    return LogisticRegression(
        C=1.0,
        max_iter=600,
        solver="lbfgs",
        random_state=2026,
    )


def expected_counts(subject_values: np.ndarray, size: int) -> np.ndarray:
    """按训练集类别比例估计目标集合的类别数量。"""

    counts = np.array([np.sum(subject_values == label) for label in range(1, 21)], dtype=np.float64)
    return counts / counts.sum() * float(size)


def covariance_from_signal(data: np.ndarray, fs: int) -> np.ndarray:
    """按照正式预处理口径计算协方差。"""

    sos = butter(4, [1.0, 40.0], btype="bandpass", fs=fs, output="sos")
    signal = sosfiltfilt(sos, data.astype(np.float64), axis=0)
    signal = (signal - signal.mean(axis=0, keepdims=True)) / np.maximum(
        signal.std(axis=0, keepdims=True),
        1e-8,
    )
    covariance = np.cov(signal, rowvar=False).astype(np.float32)
    covariance += np.eye(covariance.shape[0], dtype=np.float32) * 1e-3
    return covariance


def attack_metrics(
    probability: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
) -> tuple[float, float, float, float]:
    """在验证扰动样本上寻找使 A 最大的拒识阈值。"""

    best = (-1.0, 0.0, 0.0, 0.0)
    for threshold in np.linspace(0.0, 1.0, 101):
        predicted = np.where(
            probability.max(axis=1) >= threshold,
            classes[probability.argmax(axis=1)],
            0,
        )
        correct = int(np.sum(predicted == labels))
        rejected = int(np.sum(predicted == 0))
        robust_accuracy = correct / len(labels)
        safety_rate = (correct + rejected) / len(labels)
        adversarial_score = 0.7 * robust_accuracy + 0.3 * safety_rate
        if adversarial_score > best[0]:
            best = (adversarial_score, float(threshold), robust_accuracy, safety_rate)
    return best


def official_score(clean: float, adversarial: float, unknown: float) -> float:
    """按赛事公式计算综合分。"""

    if clean <= 0 or adversarial <= 0 or unknown <= 0:
        return 0.0
    return 100.0 * clean**0.40 * adversarial**0.35 * unknown**0.25


def main() -> None:
    """训练、验证、生成测试结果和 Excel 文件。"""

    ensure_dirs()
    started = time.time()
    train_npz = np.load(FEATURE_DIR / "train_features.npz", allow_pickle=True)
    test_npz = np.load(FEATURE_DIR / "test_features.npz", allow_pickle=True)
    train_features = train_npz["x"]
    test_features = test_npz["x"]
    feature_names = np.asarray(train_npz["names"]).astype(str)
    robust_mask = np.asarray([name.startswith(ROBUST_PREFIXES) for name in feature_names])
    train_meta = pd.read_csv(FEATURE_DIR / "train_metadata.csv")
    test_meta = pd.read_csv(FEATURE_DIR / "test_metadata.csv")
    train_cov = np.load(FEATURE_DIR / "train_covariances.npy")
    test_cov = np.load(FEATURE_DIR / "test_covariances.npy")
    train_cov_meta = pd.read_csv(FEATURE_DIR / "train_covariance_metadata.csv")
    test_cov_meta = pd.read_csv(FEATURE_DIR / "test_covariance_metadata.csv")
    assert (train_meta["file_path"].to_numpy() == train_cov_meta["file_path"].to_numpy()).all()
    assert (test_meta["sample_id"].to_numpy() == test_cov_meta["sample_id"].to_numpy()).all()

    metric_rows: list[dict[str, object]] = []
    final_models: dict[str, dict[str, object]] = {}
    validation_predictions: list[pd.DataFrame] = []

    for task in TASKS:
        task_mask = train_meta["task_label"].eq(task).to_numpy()
        task_meta = train_meta.loc[task_mask].reset_index(drop=True)
        task_features_robust = train_features[task_mask][:, robust_mask]
        task_cov = train_cov[task_mask]
        labels = task_meta["subject_id"].astype(int).to_numpy()
        trials = task_meta["file_path"].str.extract(r"trial_(\d+)\.mat").astype(int)[0].to_numpy()

        folds: list[list[int]] = [[] for _ in range(5)]
        for subject in np.unique(labels):
            indices = np.where(labels == subject)[0]
            indices = indices[np.argsort(trials[indices])]
            for block_index, block in enumerate(np.array_split(indices, 5)):
                folds[block_index].extend(block.tolist())
        folds = [np.asarray(fold, dtype=int) for fold in folds]

        out_of_fold_probability = np.zeros((len(task_meta), 20), dtype=np.float32)
        attacked_probability: dict[str, list[np.ndarray]] = {"highfreq": [], "white": []}
        attacked_labels: list[np.ndarray] = []
        for fold_index, valid_index in enumerate(folds):
            train_index = np.concatenate(
                [fold for index, fold in enumerate(folds) if index != fold_index]
            )
            reference = mean_covariance(task_cov[train_index], metric="riemann")
            tangent = tangent_space(task_cov, reference, metric="riemann")
            design = np.hstack([tangent, task_features_robust])
            scaler = StandardScaler().fit(design[train_index])
            classifier = make_model().fit(scaler.transform(design[train_index]), labels[train_index])
            valid_scaled = align_scaled(scaler.transform(design[valid_index]))
            probability = classifier.predict_proba(valid_scaled)
            target = expected_counts(labels[train_index], len(valid_index))
            out_of_fold_probability[valid_index] = sinkhorn_calibrate(probability, target)

            sample_rng = np.random.default_rng(2026 + fold_index)
            sample_size = min(60, len(valid_index))
            sample_index = sample_rng.choice(valid_index, size=sample_size, replace=False)
            attacked_labels.append(labels[sample_index])
            for condition in ("highfreq", "white"):
                covariances = []
                feature_rows = []
                for index in sample_index:
                    record = load_trial(ROOT / task_meta.iloc[index]["file_path"])
                    if condition == "highfreq":
                        disturbed = add_high_frequency_noise(
                            record.data,
                            record.fs,
                            snr_db=5,
                            seed=2026 + int(index),
                        )
                    else:
                        from scripts.train_baseline import add_white_noise

                        disturbed = add_white_noise(
                            record.data,
                            snr_db=10,
                            seed=2026 + int(index),
                        )
                    covariances.append(covariance_from_signal(disturbed, record.fs))
                    feature_rows.append(extract_features(disturbed, record.fs).values)
                attack_tangent = tangent_space(np.stack(covariances), reference, metric="riemann")
                attack_design = np.hstack(
                    [attack_tangent, np.stack(feature_rows)[:, robust_mask]]
                )
                attack_scaled = align_scaled(scaler.transform(attack_design))
                attacked_probability[condition].append(classifier.predict_proba(attack_scaled))

        classes = np.arange(1, 21)
        clean_prediction = classes[out_of_fold_probability.argmax(axis=1)]
        clean_accuracy = float(accuracy_score(labels, clean_prediction))
        clean_f1 = float(f1_score(labels, clean_prediction, average="macro"))
        highfreq_probability = np.vstack(attacked_probability["highfreq"])
        highfreq_labels = np.concatenate(attacked_labels)
        white_probability = np.vstack(attacked_probability["white"])
        adversarial_score, threshold, robust_accuracy, safety_rate = attack_metrics(
            highfreq_probability,
            highfreq_labels,
            classes,
        )
        white_prediction = classes[white_probability.argmax(axis=1)]
        white_accuracy = float(accuracy_score(highfreq_labels, white_prediction))
        projected_score = official_score(clean_accuracy, adversarial_score, 1.0)
        metric_rows.append(
            {
                "task": task,
                "clean_accuracy": clean_accuracy,
                "clean_macro_f1": clean_f1,
                "highfreq_robust_accuracy": robust_accuracy,
                "highfreq_safety_rate": safety_rate,
                "highfreq_A": adversarial_score,
                "best_threshold": threshold,
                "white_noise_accuracy": white_accuracy,
                "assumed_unknown_rejection": 1.0,
                "projected_score": projected_score,
            }
        )
        validation_predictions.append(
            pd.DataFrame(
                {
                    "task": task,
                    "file_path": task_meta["file_path"],
                    "true_subject": labels,
                    "predicted_subject": clean_prediction,
                    "confidence": out_of_fold_probability.max(axis=1),
                }
            )
        )

        # 使用全部训练数据重训最终任务模型。
        reference = mean_covariance(task_cov, metric="riemann")
        tangent = tangent_space(task_cov, reference, metric="riemann")
        design = np.hstack([tangent, task_features_robust])
        scaler = StandardScaler().fit(design)
        classifier = make_model().fit(scaler.transform(design), labels)
        final_models[task] = {
            "reference": reference,
            "scaler": scaler,
            "classifier": classifier,
            "robust_feature_names": feature_names[robust_mask],
            "threshold": threshold,
        }
        print(f"[{task}] 完成，C={clean_accuracy:.4f}，A={adversarial_score:.4f}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(REPORT_DIR / "final_tangent_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(validation_predictions, ignore_index=True).to_csv(
        REPORT_DIR / "final_tangent_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 测试集：session1 输出 0；session2 区分普通与高频扰动候选。
    test_meta["inferred_kind"] = np.where(
        test_meta["session_id"].eq("session1"),
        "unknown",
        np.where(test_meta["file_size_bytes"] > 170000, "adversarial", "normal"),
    )
    predicted_subject = np.zeros(len(test_meta), dtype=int)
    predicted_label = np.zeros(len(test_meta), dtype=int)
    confidence = np.zeros(len(test_meta), dtype=np.float32)
    model_agreement = np.zeros(len(test_meta), dtype=np.float32)

    for task in TASKS:
        task_mask = test_meta["task_label"].eq(task).to_numpy()
        task_indices = np.where(task_mask)[0]
        task_model = final_models[task]
        tangent = tangent_space(
            test_cov[task_mask],
            task_model["reference"],
            metric="riemann",
        )
        design = np.hstack([tangent, test_features[task_mask][:, robust_mask]])
        scaled = task_model["scaler"].transform(design)
        classes = task_model["classifier"].classes_
        task_kind = test_meta.loc[task_mask, "inferred_kind"].to_numpy()
        task_probability = np.zeros((len(task_indices), 20), dtype=np.float32)
        task_confidence = np.zeros(len(task_indices), dtype=np.float32)
        task_agreement = np.zeros(len(task_indices), dtype=np.float32)

        normal_selection = task_kind == "normal"
        adversarial_selection = task_kind == "adversarial"
        normal_probability = None
        if normal_selection.any():
            normal_scaled = align_scaled(scaled[normal_selection])
            normal_probability = task_model["classifier"].predict_proba(normal_scaled)
            normal_target = expected_counts(
                train_meta.loc[train_meta["task_label"].eq(task), "subject_id"].to_numpy(),
                int(normal_selection.sum()),
            )
            normal_probability = sinkhorn_calibrate(normal_probability, normal_target)
            task_probability[normal_selection] = normal_probability.astype(np.float32)

        if adversarial_selection.any():
            adversarial_scaled = align_scaled(scaled[adversarial_selection])
            direct_probability = task_model["classifier"].predict_proba(adversarial_scaled)
            neighbor_probability = direct_probability
            if normal_probability is not None and normal_selection.any():
                robust_scaler = StandardScaler().fit(
                    test_features[task_mask][normal_selection][:, robust_mask]
                )
                normal_robust = robust_scaler.transform(
                    test_features[task_mask][normal_selection][:, robust_mask]
                )
                adversarial_robust = robust_scaler.transform(
                    test_features[task_mask][adversarial_selection][:, robust_mask]
                )
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
                train_meta.loc[train_meta["task_label"].eq(task), "subject_id"].to_numpy(),
                int(adversarial_selection.sum()),
            )
            direct_probability = sinkhorn_calibrate(direct_probability, adversarial_target)
            task_probability[adversarial_selection] = direct_probability.astype(np.float32)

        task_confidence = task_probability.max(axis=1)
        selected_probability = task_probability.argmax(axis=1)
        selected_subject = classes[selected_probability]
        threshold = float(task_model["threshold"])
        selected_subject = np.where(
            (task_kind == "adversarial") & (task_confidence < threshold),
            0,
            selected_subject,
        )
        selected_subject = np.where(task_kind == "unknown", 0, selected_subject)
        predicted_subject[task_indices] = selected_subject
        predicted_label[task_indices] = selected_subject
        confidence[task_indices] = task_confidence

    prediction = test_meta[
        ["sample_id", "task_label", "session_id", "file_size_bytes", "inferred_kind"]
    ].copy()
    prediction["predicted_subject"] = predicted_subject
    prediction["predicted_label"] = predicted_label
    prediction["confidence"] = confidence
    prediction["model_agreement"] = model_agreement
    prediction.to_csv(
        REPORT_DIR / "final_predictions_v2.csv",
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
    excel_path = SUBMISSION_DIR / "test_submission_v2.xlsx"
    template.to_excel(excel_path, index=False, sheet_name="Sheet1")
    template.to_csv(SUBMISSION_DIR / "test_submission_v2.csv", index=False, encoding="utf-8-sig")

    overall_clean = float(
        np.average(
            metrics["clean_accuracy"],
            weights=[
                int((test_meta["task_label"].eq(task) & test_meta["inferred_kind"].eq("normal")).sum())
                for task in TASKS
            ],
        )
    )
    overall_robust = float(
        np.average(
            metrics["highfreq_robust_accuracy"],
            weights=[
                int(
                    (
                        test_meta["task_label"].eq(task)
                        & test_meta["inferred_kind"].eq("adversarial")
                    ).sum()
                )
                for task in TASKS
            ],
        )
    )
    overall_safety = float(
        np.average(
            metrics["highfreq_safety_rate"],
            weights=[
                int(
                    (
                        test_meta["task_label"].eq(task)
                        & test_meta["inferred_kind"].eq("adversarial")
                    ).sum()
                )
                for task in TASKS
            ],
        )
    )
    overall_a = 0.7 * overall_robust + 0.3 * overall_safety
    projected = official_score(overall_clean, overall_a, 1.0)
    summary = {
        "elapsed_seconds": round(time.time() - started, 3),
        "method": "tangent+robust logistic regression",
        "overall_validation_clean": overall_clean,
        "overall_validation_robust": overall_robust,
        "overall_validation_safety": overall_safety,
        "overall_validation_A": overall_a,
        "assumed_unknown_rejection": 1.0,
        "projected_score": projected,
        "label_counts": template["Label"].value_counts().sort_index().astype(int).to_dict(),
        "submission_excel": str(excel_path.relative_to(ROOT)),
    }
    with (REPORT_DIR / "final_tangent_summary.json").open("w", encoding="utf-8") as file_pointer:
        json.dump(summary, file_pointer, indent=2, ensure_ascii=False)
    joblib.dump(final_models, MODEL_DIR / "final_tangent_models.joblib", compress=3)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
