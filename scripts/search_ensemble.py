#!/usr/bin/env python3
"""搜索切空间与稳健特征的集成权重。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyriemann.geometry.mean import mean_covariance
from pyriemann.geometry.tangentspace import tangent_space
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from brainprint_comp.data import load_trial
from brainprint_comp.features import extract_features
from scripts.train_baseline import add_high_frequency_noise, add_white_noise, sinkhorn_calibrate


TASKS = ("EC", "EO", "Imagery", "SSVEP")
FEATURE_DIR = ROOT / "outputs" / "features"
REPORT_DIR = ROOT / "outputs" / "reports"


def build_model() -> LogisticRegression:
    """固定分类器参数。"""

    return LogisticRegression(C=1.0, max_iter=600, solver="lbfgs", random_state=2026)


def standardize_target(scaled: np.ndarray) -> np.ndarray:
    """对目标特征重新标准化。"""

    return (scaled - scaled.mean(axis=0, keepdims=True)) / np.maximum(
        scaled.std(axis=0, keepdims=True),
        1e-8,
    )


def covariance(signal: np.ndarray, fs: int) -> np.ndarray:
    """计算正则化协方差。"""

    sos = butter(4, [1.0, 40.0], btype="bandpass", fs=fs, output="sos")
    data = sosfiltfilt(sos, signal.astype(np.float64), axis=0)
    data = (data - data.mean(axis=0, keepdims=True)) / np.maximum(
        data.std(axis=0, keepdims=True),
        1e-8,
    )
    matrix = np.cov(data, rowvar=False).astype(np.float32)
    matrix += np.eye(18, dtype=np.float32) * 1e-3
    return matrix


def main() -> None:
    """搜索并输出集成结果。"""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_npz = np.load(FEATURE_DIR / "train_features.npz", allow_pickle=True)
    train_x = train_npz["x"]
    feature_names = np.asarray(train_npz["names"]).astype(str)
    train_meta = pd.read_csv(FEATURE_DIR / "train_metadata.csv")
    train_cov = np.load(FEATURE_DIR / "train_covariances.npy")
    robust_mask = np.asarray(
        [name.startswith(("relative_", "spectral_entropy_", "corr_")) for name in feature_names]
    )
    relcorr_mask = np.asarray(
        [name.startswith(("relative_", "corr_")) for name in feature_names]
    )

    rows = []
    weights = {
        "m1": {"tan_rob": 1.0},
        "m2": {"tan_all": 1.0},
        "m3": {"tan_relcorr": 1.0},
        "e1": {"tan_rob": 0.6, "tan_all": 0.4},
        "e2": {"tan_rob": 0.5, "tan_all": 0.3, "tan_relcorr": 0.2},
        "e3": {"tan_rob": 0.7, "tan_all": 0.2, "tan_relcorr": 0.1},
    }

    for task in TASKS:
        mask = train_meta["task_label"].eq(task).to_numpy()
        meta = train_meta.loc[mask].reset_index(drop=True)
        labels = meta["subject_id"].astype(int).to_numpy()
        trials = meta["file_path"].str.extract(r"trial_(\d+)\.mat").astype(int)[0].to_numpy()
        folds: list[list[int]] = [[] for _ in range(5)]
        for subject in np.unique(labels):
            indices = np.where(labels == subject)[0]
            indices = indices[np.argsort(trials[indices])]
            for block_index, block in enumerate(np.array_split(indices, 5)):
                folds[block_index].extend(block.tolist())
        folds = [np.asarray(fold, dtype=int) for fold in folds]

        design_sets = {}
        reference = mean_covariance(train_cov[mask], metric="riemann")
        tangent = tangent_space(train_cov[mask], reference, metric="riemann")
        design_sets["tan_rob"] = np.hstack([tangent, train_x[mask][:, robust_mask]])
        design_sets["tan_all"] = np.hstack([tangent, train_x[mask]])
        design_sets["tan_relcorr"] = np.hstack([tangent, train_x[mask][:, relcorr_mask]])

        clean_probability = {name: np.zeros((len(meta), 20), dtype=np.float32) for name in weights}
        attack_probability = {
            condition: {name: [] for name in weights}
            for condition in ("highfreq", "white")
        }
        attack_labels: list[np.ndarray] = []

        for fold_index, valid_index in enumerate(folds):
            train_index = np.concatenate(
                [fold for index, fold in enumerate(folds) if index != fold_index]
            )
            fold_probabilities = {}
            for design_name, design in design_sets.items():
                scaler = StandardScaler().fit(design[train_index])
                classifier = build_model().fit(
                    scaler.transform(design[train_index]),
                    labels[train_index],
                )
                fold_probabilities[design_name] = classifier.predict_proba(
                    standardize_target(scaler.transform(design[valid_index]))
                )
                target_counts = np.array(
                    [np.sum(labels[train_index] == label) for label in range(1, 21)],
                    dtype=float,
                )
            for name, mix in weights.items():
                probability = sum(
                    weight * fold_probabilities[design_name]
                    for design_name, weight in mix.items()
                )
                probability /= sum(mix.values())
                clean_probability[name][valid_index] = sinkhorn_calibrate(
                    probability,
                    target_counts / target_counts.sum() * len(valid_index),
                )

            sample_size = min(60, len(valid_index))
            sample_index = np.random.default_rng(2026 + fold_index).choice(
                valid_index,
                sample_size,
                replace=False,
            )
            attack_labels.append(labels[sample_index])
            for condition in ("highfreq", "white"):
                attack_covariance = []
                attack_feature = []
                for index in sample_index:
                    record = load_trial(ROOT / meta.iloc[index]["file_path"])
                    if condition == "highfreq":
                        disturbed = add_high_frequency_noise(
                            record.data, record.fs, 5, 2026 + int(index)
                        )
                    else:
                        disturbed = add_white_noise(
                            record.data, 10, 2026 + int(index)
                        )
                    attack_covariance.append(covariance(disturbed, record.fs))
                    attack_feature.append(extract_features(disturbed, record.fs).values)
                attack_tangent = tangent_space(
                    np.stack(attack_covariance),
                    reference,
                    metric="riemann",
                )
                attack_feature = np.stack(attack_feature)
                attack_designs = {
                    "tan_rob": np.hstack([attack_tangent, attack_feature[:, robust_mask]]),
                    "tan_all": np.hstack([attack_tangent, attack_feature]),
                    "tan_relcorr": np.hstack([attack_tangent, attack_feature[:, relcorr_mask]]),
                }
                probability_map = {}
                # 这里使用该折最后一个分类器对应的 scaler 不适合不同特征集，重训以保持口径一致。
                for design_name, design in design_sets.items():
                    scaler = StandardScaler().fit(design[train_index])
                    classifier = build_model().fit(
                        scaler.transform(design[train_index]),
                        labels[train_index],
                    )
                    probability_map[design_name] = classifier.predict_proba(
                        standardize_target(scaler.transform(attack_designs[design_name]))
                    )
                for name, mix in weights.items():
                    probability = sum(
                        weight * probability_map[design_name]
                        for design_name, weight in mix.items()
                    )
                    probability /= sum(mix.values())
                    attack_probability[condition][name].append(probability)

        classes = np.arange(1, 21)
        for name in weights:
            prediction = classes[clean_probability[name].argmax(axis=1)]
            clean_accuracy = accuracy_score(labels, prediction)
            highfreq_p = np.vstack(attack_probability["highfreq"][name])
            white_p = np.vstack(attack_probability["white"][name])
            attack_y = np.concatenate(attack_labels)
            highfreq_accuracy = accuracy_score(
                attack_y,
                classes[highfreq_p.argmax(axis=1)],
            )
            white_accuracy = accuracy_score(
                attack_y,
                classes[white_p.argmax(axis=1)],
            )
            best_a = -1.0
            best_threshold = 0.0
            best_ra = 0.0
            best_safety = 0.0
            for threshold in np.linspace(0.0, 1.0, 101):
                predicted = np.where(
                    highfreq_p.max(axis=1) >= threshold,
                    classes[highfreq_p.argmax(axis=1)],
                    0,
                )
                correct = int(np.sum(predicted == attack_y))
                rejected = int(np.sum(predicted == 0))
                robust_accuracy = correct / len(attack_y)
                safety_rate = (correct + rejected) / len(attack_y)
                adversarial_score = 0.7 * robust_accuracy + 0.3 * safety_rate
                if adversarial_score > best_a:
                    best_a = adversarial_score
                    best_threshold = threshold
                    best_ra = robust_accuracy
                    best_safety = safety_rate
            rows.append(
                {
                    "task": task,
                    "ensemble": name,
                    "clean_accuracy": clean_accuracy,
                    "highfreq_accuracy": highfreq_accuracy,
                    "white_accuracy": white_accuracy,
                    "clean_plus_highfreq": clean_accuracy + highfreq_accuracy,
                    "best_threshold": best_threshold,
                    "best_RA": best_ra,
                    "best_ASafeR": best_safety,
                    "best_A": best_a,
                }
            )

    results = pd.DataFrame(rows)
    results.to_csv(
        REPORT_DIR / "ensemble_search.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(results.sort_values(["task", "clean_plus_highfreq"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
