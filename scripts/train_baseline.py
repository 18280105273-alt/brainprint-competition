#!/usr/bin/env python3
"""训练初版跨会话身份识别基线并生成测试预测。"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brainprint_comp.data import load_trial
from brainprint_comp.features import extract_features
from brainprint_comp.modeling import fit_task_model, predict_proba, select_features


TASKS = ("EC", "EO", "Imagery", "SSVEP")
MODEL_DIR = ROOT / "outputs" / "models"
REPORT_DIR = ROOT / "outputs" / "reports"
FIGURE_DIR = ROOT / "outputs" / "figures"
SUBMISSION_DIR = ROOT / "outputs" / "submissions"


@dataclass
class TemporalModels:
    """保存一个任务的全部特征模型和稳健特征模型。"""

    all_features: object
    robust_features: object
    train_idx: np.ndarray
    valid_idx: np.ndarray


def ensure_directories() -> None:
    """建立输出目录。"""

    for directory in (MODEL_DIR, REPORT_DIR, FIGURE_DIR, SUBMISSION_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def temporal_split(task_meta: pd.DataFrame, train_fraction: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """按受试者和 trial 序号切分前段训练、后段验证。"""

    trial = task_meta["file_path"].str.extract(r"trial_(\d+)\.mat").astype(int)[0].to_numpy()
    subject = task_meta["subject_id"].to_numpy()
    train_indices: list[int] = []
    valid_indices: list[int] = []
    for subject_id in np.unique(subject):
        indices = np.where(subject == subject_id)[0]
        indices = indices[np.argsort(trial[indices])]
        cut = max(1, int(len(indices) * train_fraction))
        train_indices.extend(indices[:cut])
        valid_indices.extend(indices[cut:])
    return np.asarray(train_indices), np.asarray(valid_indices)


def add_high_frequency_noise(data: np.ndarray, fs: int, snr_db: float, seed: int) -> np.ndarray:
    """按照指定 SNR 加入 45 Hz 以上高频噪声。"""

    rng = np.random.default_rng(seed)
    noise = rng.normal(size=data.shape)
    high = min(120.0, fs / 2 - 1)
    sos = butter(4, [45.0, high], btype="bandpass", fs=fs, output="sos")
    noise = sosfiltfilt(sos, noise, axis=0)
    signal_rms = np.sqrt(np.mean(data**2))
    noise_rms = np.sqrt(np.mean(noise**2))
    target_rms = signal_rms * 10 ** (-snr_db / 20)
    return data + noise * (target_rms / max(noise_rms, 1e-12))


def add_white_noise(data: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """加入宽带高斯白噪声。"""

    rng = np.random.default_rng(seed)
    noise = rng.normal(size=data.shape)
    signal_rms = np.sqrt(np.mean(data**2))
    noise_rms = np.sqrt(np.mean(noise**2))
    target_rms = signal_rms * 10 ** (-snr_db / 20)
    return data + noise * (target_rms / max(noise_rms, 1e-12))


def channel_dropout(data: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """随机将部分通道置零。"""

    rng = np.random.default_rng(seed)
    output = data.copy()
    count = max(1, int(round(data.shape[1] * fraction)))
    channels = rng.choice(data.shape[1], size=count, replace=False)
    output[:, channels] = 0.0
    return output


def sinkhorn_calibrate(probability: np.ndarray, target_counts: np.ndarray, iterations: int = 20) -> np.ndarray:
    """在无标签测试集上按预期类别数量做软校准。"""

    calibrated = probability.copy()
    target_counts = np.asarray(target_counts, dtype=np.float64)
    for _ in range(iterations):
        column_sum = calibrated.sum(axis=0)
        calibrated *= np.divide(
            target_counts,
            np.maximum(column_sum, 1e-12),
            out=np.zeros_like(target_counts),
            where=column_sum > 0,
        )
        row_sum = calibrated.sum(axis=1, keepdims=True)
        calibrated /= np.maximum(row_sum, 1e-12)
    return calibrated


def predict_aligned(model, features: np.ndarray, feature_names: np.ndarray) -> np.ndarray:
    """把目标会话特征逐维对齐到训练分布后再分类。"""

    mask = np.isin(feature_names, model.feature_names)
    scaler = model.estimator.named_steps["scale"]
    classifier = model.estimator.named_steps["model"]
    scaled = scaler.transform(features[:, mask])
    scaled = (scaled - scaled.mean(axis=0, keepdims=True)) / np.maximum(
        scaled.std(axis=0, keepdims=True),
        1e-8,
    )
    return classifier.predict_proba(scaled)


def evaluate_models(
    task_models: TemporalModels,
    task_meta: pd.DataFrame,
    task_x: np.ndarray,
    feature_names: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """评估普通特征模型和高频扰动鲁棒模型。"""

    valid_meta = task_meta.iloc[task_models.valid_idx].reset_index(drop=True)
    valid_x = task_x[task_models.valid_idx]
    labels = valid_meta["subject_id"].astype(int).to_numpy()

    probability_all = predict_proba(task_models.all_features, valid_x, feature_names)
    probability_robust = predict_proba(task_models.robust_features, valid_x, feature_names)
    classes = np.asarray(task_models.all_features.classes)

    rows = []
    confusion_parts = []
    for condition, matrix in (("clean", valid_x),):
        probability = matrix
        prediction_all = classes[probability_all.argmax(axis=1)]
        prediction_robust = classes[probability_robust.argmax(axis=1)]
        prediction_ensemble = classes[(0.7 * probability_all + 0.3 * probability_robust).argmax(axis=1)]
        target_counts = np.array([np.sum(labels == label) for label in classes], dtype=np.float64)
        probability_prior = sinkhorn_calibrate(
            0.7 * probability_all + 0.3 * probability_robust,
            target_counts,
        )
        prediction_prior = classes[probability_prior.argmax(axis=1)]
        for model_name, prediction in (
            ("all", prediction_all),
            ("robust", prediction_robust),
            ("ensemble", prediction_ensemble),
            ("ensemble_prior", prediction_prior),
        ):
            rows.append(
                {
                    "task": task_models.all_features.task,
                    "condition": condition,
                    "model": model_name,
                    "accuracy": accuracy_score(labels, prediction),
                    "macro_f1": f1_score(labels, prediction, average="macro"),
                }
            )
        confusion_parts.append(prediction_prior)

    # 使用验证集前 300 条构造可控干扰，控制特征提取耗时。
    sample_count = min(300, len(valid_meta))
    sample_index = np.linspace(0, len(valid_meta) - 1, sample_count, dtype=int)
    attacked_features: dict[str, np.ndarray] = {}
    for condition in ("highfreq_5db", "white_10db", "channel_dropout_20"):
        attacked = []
        for index in sample_index:
            record = load_trial(ROOT / valid_meta.iloc[index]["file_path"])
            data = record.data
            if condition == "highfreq_5db":
                data = add_high_frequency_noise(data, record.fs, snr_db=5, seed=2026 + index)
            elif condition == "white_10db":
                data = add_white_noise(data, snr_db=10, seed=2026 + index)
            elif condition == "channel_dropout_20":
                data = channel_dropout(data, fraction=0.2, seed=2026 + index)
            attacked.append(extract_features(data, record.fs).values)
        attacked_features[condition] = np.stack(attacked)

    sample_labels = labels[sample_index]
    for condition, matrix in attacked_features.items():
        probability_all = predict_proba(task_models.all_features, matrix, feature_names)
        probability_robust = predict_proba(task_models.robust_features, matrix, feature_names)
        prediction_all = classes[probability_all.argmax(axis=1)]
        prediction_robust = classes[probability_robust.argmax(axis=1)]
        prediction_ensemble = classes[(0.2 * probability_all + 0.8 * probability_robust).argmax(axis=1)]
        for model_name, prediction in (
            ("all", prediction_all),
            ("robust", prediction_robust),
            ("ensemble", prediction_ensemble),
        ):
            rows.append(
                {
                    "task": task_models.all_features.task,
                    "condition": condition,
                    "model": model_name,
                    "accuracy": accuracy_score(sample_labels, prediction),
                    "macro_f1": f1_score(sample_labels, prediction, average="macro"),
                }
            )

    return pd.DataFrame(rows), valid_meta, labels, np.asarray(confusion_parts[0])


def main() -> None:
    """训练、评估并生成初版提交。"""

    ensure_directories()
    started = time.time()
    train_npz = np.load(ROOT / "outputs" / "features" / "train_features.npz", allow_pickle=True)
    test_npz = np.load(ROOT / "outputs" / "features" / "test_features.npz", allow_pickle=True)
    features = train_npz["x"]
    test_features = test_npz["x"]
    feature_names = np.asarray(train_npz["names"]).astype(str)
    train_meta = pd.read_csv(ROOT / "outputs" / "features" / "train_metadata.csv")
    test_meta = pd.read_csv(ROOT / "outputs" / "features" / "test_metadata.csv")

    metrics_frames = []
    confusion_rows = []
    final_models: dict[str, dict[str, object]] = {}

    for task in TASKS:
        mask = train_meta["task_label"].eq(task).to_numpy()
        task_x = features[mask]
        task_meta = train_meta.loc[mask].reset_index(drop=True)
        labels = task_meta["subject_id"].astype(int).to_numpy()
        train_idx, valid_idx = temporal_split(task_meta)
        temporal = TemporalModels(
            all_features=fit_task_model(
                task,
                task_x[train_idx],
                labels[train_idx],
                feature_names,
                "all",
            ),
            robust_features=fit_task_model(
                task,
                task_x[train_idx],
                labels[train_idx],
                feature_names,
                "robust",
            ),
            train_idx=train_idx,
            valid_idx=valid_idx,
        )
        metrics, valid_meta, true_labels, predictions = evaluate_models(
            temporal,
            task_meta,
            task_x,
            feature_names,
        )
        metrics_frames.append(metrics)
        for true_label, predicted_label in zip(true_labels, predictions):
            confusion_rows.append(
                {"task": task, "true_subject": int(true_label), "predicted_subject": int(predicted_label)}
            )

        # 用全部训练数据重训最终模型。
        final_models[task] = {
            "all": fit_task_model(task, task_x, labels, feature_names, "all"),
            "robust": fit_task_model(task, task_x, labels, feature_names, "robust"),
        }
        print(f"[{task}] 时间顺序验证完成", flush=True)

    metrics = pd.concat(metrics_frames, ignore_index=True)
    metrics.to_csv(REPORT_DIR / "baseline_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(confusion_rows).to_csv(
        REPORT_DIR / "temporal_confusion_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 测试集预测：session1 视为未知身份；session2 根据文件大小区分普通与高频扰动候选。
    test_meta["inferred_kind"] = np.where(
        test_meta["session_id"].eq("session1"),
        "unknown",
        np.where(test_meta["file_size_bytes"] > 170000, "adversarial", "normal"),
    )
    probability = np.zeros((len(test_meta), 20), dtype=np.float32)
    model_agreement = np.zeros(len(test_meta), dtype=np.float32)
    for task in TASKS:
        mask = test_meta["task_label"].eq(task).to_numpy()
        task_x = test_features[mask]
        kind = test_meta.loc[mask, "inferred_kind"].to_numpy()
        task_probability = np.zeros((len(task_x), 20), dtype=np.float32)
        task_agreement = np.zeros(len(task_x), dtype=np.float32)
        task_train_meta = train_meta.loc[train_meta["task_label"].eq(task)]
        train_subject_counts = np.asarray(
            [np.sum(task_train_meta["subject_id"].astype(int).to_numpy() == label) for label in range(1, 21)],
            dtype=np.float64,
        )
        for kind_name in ("normal", "adversarial"):
            selection = kind == kind_name
            if not selection.any():
                continue
            probability_all = predict_aligned(
                final_models[task]["all"],
                task_x[selection],
                feature_names,
            )
            probability_robust = predict_aligned(
                final_models[task]["robust"],
                task_x[selection],
                feature_names,
            )
            if kind_name == "adversarial":
                weighted = 0.2 * probability_all + 0.8 * probability_robust
            else:
                weighted = 0.7 * probability_all + 0.3 * probability_robust
            target_counts = train_subject_counts / train_subject_counts.sum() * float(selection.sum())
            task_probability[selection] = sinkhorn_calibrate(weighted, target_counts).astype(np.float32)
            task_agreement[selection] = (
                probability_all.argmax(axis=1) == probability_robust.argmax(axis=1)
            ).astype(np.float32)
        probability[mask] = task_probability
        model_agreement[mask] = task_agreement

    classes = np.arange(1, 21)
    predicted_subject = classes[probability.argmax(axis=1)]
    confidence = probability.max(axis=1)
    predicted_label = np.where(test_meta["inferred_kind"].eq("unknown"), 0, predicted_subject)

    prediction_frame = test_meta[
        ["sample_id", "task_label", "session_id", "file_size_bytes", "inferred_kind"]
    ].copy()
    prediction_frame["predicted_subject"] = predicted_subject
    prediction_frame["predicted_label"] = predicted_label
    prediction_frame["confidence"] = confidence
    prediction_frame["model_agreement"] = model_agreement
    prediction_frame.sort_values("sample_id").to_csv(
        REPORT_DIR / "test_predictions_v1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    template = pd.read_csv(ROOT / "Brainprint_Competition" / "test.csv", dtype={"Sample": str})
    mapping = prediction_frame.set_index("sample_id")["predicted_label"]
    template["Label"] = template["Sample"].str.extract(r"(\d+)")[0].astype(int).map(mapping).astype(int)
    template.to_csv(SUBMISSION_DIR / "test_submission_v1.csv", index=False, encoding="utf-8-sig")

    temporal_accuracy = (
        metrics.groupby(["condition", "model"])["accuracy"]
        .mean()
        .round(4)
        .rename(index=lambda item: str(item))
    )
    summary = {
        "elapsed_seconds": round(time.time() - started, 3),
        "temporal_validation": {
            f"{condition}|{model_name}": float(value)
            for (condition, model_name), value in temporal_accuracy.items()
        },
        "submission_counts": template["Label"].value_counts().sort_index().astype(int).to_dict(),
        "test_confidence_quantiles": {
            kind: group["confidence"].quantile([0.1, 0.5, 0.9]).round(4).to_dict()
            for kind, group in prediction_frame.groupby("inferred_kind")
        },
        "model_agreement": {
            kind: round(float(group["model_agreement"].mean()), 4)
            for kind, group in prediction_frame.groupby("inferred_kind")
        },
    }
    with (REPORT_DIR / "baseline_summary.json").open("w", encoding="utf-8") as file_pointer:
        json.dump(summary, file_pointer, indent=2, ensure_ascii=False)

    joblib.dump(final_models, MODEL_DIR / "baseline_task_models.joblib", compress=3)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
