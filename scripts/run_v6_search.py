#!/usr/bin/env python3
"""第三次提交的跨会话候选搜索与最终候选生成。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from pyriemann.geometry.mean import mean_covariance
from pyriemann.geometry.tangentspace import tangent_space
from pyriemann.transfer import TLCenter, TLScale, encode_domains
from scipy.linalg import expm
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from brainprint_comp.data import load_trial
from brainprint_comp.features import extract_features
from scripts.train_baseline import sinkhorn_calibrate
from scripts.train_final_tangent import covariance_from_signal


TASKS = ("EC", "EO", "Imagery", "SSVEP")
FEATURE_DIR = ROOT / "outputs" / "features"
REPORT_DIR = ROOT / "outputs" / "reports"
SUBMISSION_DIR = ROOT / "outputs" / "submissions"
PAIRING_PATH = REPORT_DIR / "adv_clean_pairs.csv"
ROBUST_PREFIXES = ("relative_", "spectral_entropy_", "corr_")
RELCORR_PREFIXES = ("relative_", "corr_")
CLASSES = np.arange(1, 21, dtype=int)


@dataclass(frozen=True)
class Candidate:
    """一个可复现的候选配置。"""

    name: str
    view: str
    align: str
    classifier: str
    target_center: bool
    c: float = 1.0


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "predict", "finalize"], default="validate")
    parser.add_argument("--candidate", default="", help="用逗号分隔的候选名")
    parser.add_argument(
        "--calibration",
        choices=["none", "train03", "uniform03"],
        default="train03",
    )
    parser.add_argument("--reject-threshold", type=float, default=0.30)
    parser.add_argument(
        "--weights",
        default="",
        help="候选权重，形如 candidate:2,candidate:1",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--fold-mode", choices=["balanced", "contiguous"], default="balanced")
    parser.add_argument("--angles", default="0,5,10,15", help="逗号分隔的通道混合角")
    parser.add_argument("--workers", type=int, default=min(8, (Path("/").exists() and 8) or 1))
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "v6_search_summary.json")
    return parser.parse_args()


def load_data() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
]:
    """加载已有特征和协方差缓存。"""

    train = np.load(FEATURE_DIR / "train_features.npz", allow_pickle=True)
    test = np.load(FEATURE_DIR / "test_features.npz", allow_pickle=True)
    train_meta = pd.read_csv(FEATURE_DIR / "train_metadata.csv")
    test_meta = pd.read_csv(FEATURE_DIR / "test_metadata.csv")
    train_cov = np.load(FEATURE_DIR / "train_covariances.npy")
    test_cov = np.load(FEATURE_DIR / "test_covariances.npy")
    train_band = np.load(FEATURE_DIR / "train_band_covariances.npy")
    test_band = np.load(FEATURE_DIR / "test_band_covariances.npy")
    pairing = pd.read_csv(PAIRING_PATH)
    return (
        train["x"],
        np.asarray(train["names"]).astype(str),
        train_meta,
        train_cov,
        train_band,
        test["x"],
        test_meta,
        test_cov,
        test_band,
        pairing,
    )


def view_mask(names: np.ndarray, view: str) -> np.ndarray:
    """选择手工特征列。"""

    if view == "all":
        return np.ones(len(names), dtype=bool)
    if view == "robust":
        return np.asarray([name.startswith(ROBUST_PREFIXES) for name in names])
    if view == "relcorr":
        return np.asarray([name.startswith(RELCORR_PREFIXES) for name in names])
    if view in {"none", "tangent"}:
        return np.zeros(len(names), dtype=bool)
    raise ValueError(f"未知特征视图: {view}")


def make_classifier(name: str, c: float) -> object:
    """构造分类器或返回 None，表示使用黎曼原型距离。"""

    if name == "lr":
        return LogisticRegression(C=c, max_iter=1000, solver="lbfgs", random_state=2026)
    if name == "lda":
        return LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr")
    if name == "nc":
        return NearestCentroid(shrink_threshold=None)
    if name == "mdm":
        return None
    raise ValueError(f"未知分类器: {name}")


def tangent_from_cov(
    cov: np.ndarray,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """在给定或默认参考处切空间化。"""

    if reference is None:
        reference = mean_covariance(cov, metric="riemann")
    return tangent_space(cov, reference, metric="riemann")


def band_tangent_from_cov(
    band_cov: np.ndarray,
    reference_per_band: list[np.ndarray] | None = None,
) -> np.ndarray:
    """把三个频带协方差分别切空间化后拼接。"""

    if reference_per_band is None:
        reference_per_band = [
            mean_covariance(band_cov[:, index], metric="riemann")
            for index in range(band_cov.shape[1])
        ]
    parts = [
        tangent_space(band_cov[:, index], reference_per_band[index], metric="riemann")
        for index in range(band_cov.shape[1])
    ]
    return np.hstack(parts)


def align_covariances(
    source_cov: np.ndarray,
    target_cov: np.ndarray,
    source_labels: np.ndarray,
    align: str,
) -> tuple[np.ndarray, np.ndarray]:
    """执行协方差域对齐并返回源、目标切空间。"""

    all_cov = np.concatenate([source_cov, target_cov])
    domains = np.concatenate(
        [np.zeros(len(source_cov), dtype=int), np.ones(len(target_cov), dtype=int)]
    )
    encoded_labels = np.concatenate(
        [source_labels, np.zeros(len(target_cov), dtype=int)]
    )

    if align == "none":
        reference = mean_covariance(source_cov, metric="riemann")
        tangent = tangent_from_cov(all_cov, reference)
    elif align in {"tlc", "tlc_scale"}:
        encoded_x, encoded_y = encode_domains(all_cov, encoded_labels, domains)
        centered = TLCenter(target_domain=1, metric="riemann").fit_transform(
            encoded_x, encoded_y
        )
        if align == "tlc_scale":
            centered = TLScale(target_domain=1, metric="riemann").fit_transform(
                centered, encoded_y
            )
        reference = mean_covariance(centered[: len(source_cov)], metric="riemann")
        tangent = tangent_from_cov(centered, reference)
    else:
        raise ValueError(f"未知对齐方式: {align}")
    return tangent[: len(source_cov)], tangent[len(source_cov) :]


def align_band_covariances(
    source_band_cov: np.ndarray,
    target_band_cov: np.ndarray,
    source_labels: np.ndarray,
    align: str,
) -> tuple[np.ndarray, np.ndarray]:
    """逐频带对齐协方差并拼接为 source/target × (bands * tangent_dim)。"""

    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for index in range(source_band_cov.shape[1]):
        source_tangent, target_tangent = align_covariances(
            source_band_cov[:, index],
            target_band_cov[:, index],
            source_labels,
            align,
        )
        source_parts.append(source_tangent)
        target_parts.append(target_tangent)
    return np.hstack(source_parts), np.hstack(target_parts)


def build_design(
    task: str,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    features: np.ndarray,
    names: np.ndarray,
    covariance: np.ndarray,
    band_covariance: np.ndarray,
    labels: np.ndarray,
    candidate: Candidate,
    target_features: np.ndarray | None = None,
    target_covariance: np.ndarray | None = None,
    target_band_covariance: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造指定候选的源、目标设计矩阵和标签。"""

    source_feature = features[source_indices]
    target_feature = (
        target_features
        if target_features is not None
        else features[target_indices]
    )
    source_cov = covariance[source_indices]
    target_cov = (
        target_covariance
        if target_covariance is not None
        else covariance[target_indices]
    )
    source_band = band_covariance[source_indices]
    target_band = (
        target_band_covariance
        if target_band_covariance is not None
        else band_covariance[target_indices]
    )
    hand_view = (
        candidate.view[len("band_") :]
        if candidate.view.startswith("band_")
        else candidate.view
    )
    mask = view_mask(names, hand_view)

    if candidate.classifier == "mdm":
        return source_cov, target_cov, labels[source_indices], labels[target_indices]

    source_tangent, target_tangent = align_covariances(
        source_cov, target_cov, labels[source_indices], candidate.align
    )
    if candidate.view.startswith("band_"):
        source_band_tangent, target_band_tangent = align_band_covariances(
            source_band,
            target_band,
            labels[source_indices],
            candidate.align,
        )
    else:
        source_band_tangent = target_band_tangent = None

    parts = {
        "tan": (source_tangent, target_tangent),
        "band_tan": (
            source_band_tangent if source_band_tangent is not None else source_tangent,
            target_band_tangent if target_band_tangent is not None else target_tangent,
        ),
        "hand": (source_feature[:, mask], target_feature[:, mask]),
    }
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    if candidate.view.startswith("band_"):
        source_parts.extend([parts["band_tan"][0], parts["tan"][0]])
        target_parts.extend([parts["band_tan"][1], parts["tan"][1]])
        hand_view = candidate.view[len("band_") :]
        if hand_view != "none":
            hand_mask = view_mask(names, hand_view)
            source_parts.append(source_feature[:, hand_mask])
            target_parts.append(target_feature[:, hand_mask])
    else:
        source_parts.append(parts["tan"][0])
        target_parts.append(parts["tan"][1])
        if candidate.view not in {"none", "tangent"}:
            source_parts.append(parts["hand"][0])
            target_parts.append(parts["hand"][1])

    return (
        np.hstack(source_parts),
        np.hstack(target_parts),
        labels[source_indices],
        np.zeros(len(target_indices), dtype=int),
    )


def fit_predict_proba(
    train_design: np.ndarray,
    target_design: np.ndarray,
    train_labels: np.ndarray,
    candidate: Candidate,
    target_center: bool = False,
) -> np.ndarray:
    """训练分类器并对目标设计给出 20 类概率。"""

    if candidate.classifier == "mdm":
        from pyriemann.classification import MDM

        model = MDM(metric="riemann").fit(train_design, train_labels)
        distances = np.stack(
            [
                -np.asarray(model.transform(target_design))[:, index]
                for index in range(20)
            ],
            axis=1,
        )
        probability = np.exp(
            distances - distances.max(axis=1, keepdims=True)
        )
        probability /= probability.sum(axis=1, keepdims=True)
        return probability

    if candidate.classifier == "nc":
        estimator = make_classifier(candidate.classifier, candidate.c)
        estimator.fit(train_design, train_labels)
        distances = np.linalg.norm(
            target_design[:, None, :] - estimator.centroids_[None, :, :],
            axis=2,
        )
        probability = np.exp(-distances)
        probability /= probability.sum(axis=1, keepdims=True)
        return probability[:, np.argsort(np.asarray(estimator.classes_))]

    scaler = StandardScaler().fit(train_design)
    train_scaled = scaler.transform(train_design)
    target_scaled = scaler.transform(target_design)
    if target_center:
        tangent_dimensions = 0
        if candidate.view not in {"none", "tangent"}:
            tangent_dimensions = 171
            if candidate.view.startswith("band_"):
                tangent_dimensions = 171 * 2
        target_scaled[:, tangent_dimensions:] = (
            target_scaled[:, tangent_dimensions:]
            - target_scaled[:, tangent_dimensions:].mean(axis=0, keepdims=True)
        )
    estimator = make_classifier(candidate.classifier, candidate.c)
    estimator.fit(train_scaled, train_labels)

    if hasattr(estimator, "predict_proba"):
        probability = estimator.predict_proba(target_scaled)
    elif hasattr(estimator, "decision_function"):
        decision = estimator.decision_function(target_scaled)
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])
        probability = np.exp(
            decision - decision.max(axis=1, keepdims=True)
        )
        probability /= probability.sum(axis=1, keepdims=True)
    else:
        raise TypeError("分类器不支持概率或决策分数")
    probability = probability[:, np.argsort(np.asarray(estimator.classes_))]
    return probability


def calibrated_probability(
    probability: np.ndarray,
    train_labels: np.ndarray,
    mode: str,
) -> np.ndarray:
    """应用无校准、训练先验软校准或共享先验软校准。"""

    if mode == "none":
        return probability
    if mode == "train03":
        counts = np.asarray(
            [np.sum(train_labels == label) for label in CLASSES], dtype=float
        )
        target_counts = counts / counts.sum() * len(probability)
        calibrated = sinkhorn_calibrate(probability, target_counts)
        return 0.7 * probability + 0.3 * calibrated
    if mode == "train_full":
        counts = np.asarray(
            [np.sum(train_labels == label) for label in CLASSES], dtype=float
        )
        target_counts = counts / counts.sum() * len(probability)
        return sinkhorn_calibrate(probability, target_counts)
    if mode == "uniform03":
        calibrated = sinkhorn_calibrate(
            probability, np.full(20, len(probability) / 20.0)
        )
        return 0.7 * probability + 0.3 * calibrated
    if mode == "uniform_full":
        return sinkhorn_calibrate(
            probability, np.full(20, len(probability) / 20.0)
        )
    raise ValueError(f"未知校准模式: {mode}")


def temporal_folds(meta: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """按每位受试者的连续 trial 顺序生成五折时间分段。"""

    trials = meta["file_path"].str.extract(r"trial_(\d+)\.mat").astype(int)[0].to_numpy()
    subjects = meta["subject_id"].to_numpy()
    folds: list[list[int]] = [[] for _ in range(5)]
    for subject in np.unique(subjects):
        indices = np.where(subjects == subject)[0]
        indices = indices[np.argsort(trials[indices])]
        for fold_index, block in enumerate(np.array_split(indices, 5)):
            folds[fold_index].extend(block.tolist())
    return [
        (
            np.concatenate(
                [fold for index, fold in enumerate(folds) if index != fold_index]
            ),
            np.asarray(folds[fold_index]),
        )
        for fold_index in range(5)
    ]


def balanced_fold(meta: pd.DataFrame, fold_index: int) -> tuple[np.ndarray, np.ndarray]:
    """按受试者随机留出 20%，避免同一刺激条件批次主导目标折。"""

    rng = np.random.default_rng(2026 + fold_index)
    subjects = meta["subject_id"].to_numpy()
    target_indices: list[int] = []
    for subject in np.unique(subjects):
        indices = np.where(subjects == subject)[0]
        target_count = max(1, int(round(len(indices) * 0.2)))
        selected = rng.choice(indices, size=target_count, replace=False)
        target_indices.extend(selected.tolist())
    target = np.asarray(target_indices)
    source = np.setdiff1d(np.arange(len(meta)), target)
    return source, target


def channel_mixing_angle(angle_degrees: float, seed: int = 2026) -> np.ndarray:
    """生成接近单位阵的随机正交通道混合矩阵。"""

    rng = np.random.default_rng(seed)
    dimension = 18
    skew = rng.normal(size=(dimension, dimension))
    skew = (skew - skew.T) / 2
    return expm(np.deg2rad(angle_degrees) * skew)


def transform_trial(record, mixing: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """对单条 trial 应用通道混合和逐通道增益。"""

    data = record.data.copy()
    data = data @ mixing.T
    data *= gain[None, :]
    return data


def recompute_target_materials(
    paths: Iterable[str],
    mixing: np.ndarray,
    gain: np.ndarray,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从原始数据重算目标特征、协方差和频带协方差。"""

    paths = list(paths)
    inputs = [ROOT / path for path in paths]
    band_edges = ((4.0, 8.0), (8.0, 13.0), (13.0, 30.0))

    def compute(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        record = load_trial(path)
        data = transform_trial(record, mixing, gain)
        covariance = covariance_from_signal(data, record.fs)
        features = extract_features(data, record.fs).values
        spectrum = np.fft.rfft(
            (data - data.mean(axis=0, keepdims=True))
            / np.maximum(data.std(axis=0, keepdims=True), 1e-8),
            axis=0,
        )
        frequencies = np.fft.rfftfreq(len(data), d=1.0 / record.fs)
        bands = []
        for low, high in band_edges:
            selected = (frequencies >= low) & (frequencies < high)
            matrix = np.einsum(
                "fc,fd->cd", spectrum[selected].real, spectrum[selected].real
            )
            matrix += np.einsum(
                "fc,fd->cd", spectrum[selected].imag, spectrum[selected].imag
            )
            matrix /= max(float(np.trace(matrix)) / matrix.shape[0], 1e-12)
            matrix += np.eye(matrix.shape[0]) * 1e-3
            bands.append(matrix.astype(np.float32))
        return features, covariance, np.stack(bands)

    features: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    band_covariances: list[np.ndarray] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(compute, path): index for index, path in enumerate(inputs)}
        for future in as_completed(futures):
            feature_row, covariance_row, band_row = future.result()
            features.append((futures[future], feature_row))
            covariances.append((futures[future], covariance_row))
            band_covariances.append((futures[future], band_row))
    features.sort(key=lambda item: item[0])
    covariances.sort(key=lambda item: item[0])
    band_covariances.sort(key=lambda item: item[0])
    return (
        np.stack([item[1] for item in features]),
        np.stack([item[1] for item in covariances]),
        np.stack([item[1] for item in band_covariances]),
    )


def evaluate_candidate(
    candidate: Candidate,
    fold_index: int,
    fold_mode: str,
    angles: list[float],
    workers: int,
) -> dict[str, object]:
    """在时间分段目标折和通道混合条件下评估候选。"""

    (
        train_features,
        names,
        train_meta,
        train_cov,
        train_band,
        _,
        _,
        _,
        _,
        _,
    ) = load_data()
    task_rows: list[dict[str, object]] = []

    for task in TASKS:
        task_meta = train_meta[train_meta["task_label"].eq(task)].reset_index(drop=True)
        labels = task_meta["subject_id"].astype(int).to_numpy()
        source_fold, target_fold = (
            temporal_folds(task_meta)[fold_index]
            if fold_mode == "contiguous"
            else balanced_fold(task_meta, fold_index)
        )
        source_labels = labels[source_fold]

        for angle in angles:
            if angle == 0:
                global_target_indices = task_meta.index.to_numpy()[target_fold]
                target_features = train_features[global_target_indices]
                target_cov = train_cov[global_target_indices]
                target_band = train_band[global_target_indices]
            else:
                mixing = channel_mixing_angle(angle)
                gain = 1.0 + 0.08 * np.linspace(-1, 1, 18)
                target_features, target_cov, target_band = recompute_target_materials(
                    task_meta.loc[target_fold, "file_path"],
                    mixing,
                    gain,
                    workers,
                )

            train_design, target_design, train_labels, _ = build_design(
                task,
                source_fold,
                np.arange(len(target_fold)),
                train_features[task_meta.index.to_numpy()],
                names,
                train_cov[task_meta.index.to_numpy()],
                train_band[task_meta.index.to_numpy()],
                labels,
                candidate,
                target_features,
                target_cov,
                target_band,
            )
            probability = fit_predict_proba(
                train_design,
                target_design,
                train_labels,
                candidate,
                candidate.target_center,
            )
            for calibration in (
                "none",
                "train03",
                "uniform03",
                "train_full",
                "uniform_full",
            ):
                calibrated = calibrated_probability(
                    probability, train_labels, calibration
                )
                prediction = CLASSES[calibrated.argmax(axis=1)]
                accuracy = float(np.mean(prediction == labels[target_fold]))
                task_rows.append(
                    {
                        "task": task,
                        "angle": angle,
                        "calibration": calibration,
                        "accuracy": accuracy,
                    }
                )

    frame = pd.DataFrame(task_rows)
    summary = {
        "candidate": candidate.name,
        "fold": fold_index,
        "angles": angles,
        "accuracy": frame.groupby("calibration")["accuracy"].mean().to_dict(),
        "by_task": frame.pivot_table(
            index="task", columns="calibration", values="accuracy", aggfunc="mean"
        ).round(5).to_dict(),
        "by_angle_calibration": frame.pivot_table(
            index="angle", columns="calibration", values="accuracy", aggfunc="mean"
        ).round(5).to_dict(),
    }
    return summary


def build_candidates() -> list[Candidate]:
    """构建首轮经典候选。"""

    candidates = [
        Candidate("v5i_like", "robust", "tlc", "lr", True),
        Candidate("tlc_raw", "robust", "tlc", "lr", False),
        Candidate("tlc_scale", "robust", "tlc_scale", "lr", True),
        Candidate("tan_only", "none", "tlc", "lr", False),
        Candidate("tan_all", "all", "tlc", "lr", False),
        Candidate("band_tan_robust", "band_robust", "tlc", "lr", True),
        Candidate("lda_tan_robust", "robust", "tlc", "lda", True),
        Candidate("nc_tan_robust", "robust", "tlc", "nc", True),
        Candidate("mdm_tlc", "none", "tlc", "mdm", False),
    ]
    for c_value in (0.1, 0.3, 3.0):
        candidates.append(
            Candidate(f"lr_{c_value}", "robust", "tlc", "lr", True, c=c_value)
        )
    return candidates


def run_validate(args: argparse.Namespace) -> int:
    """运行候选验证。"""

    angles = [float(value) for value in args.angles.split(",")]
    candidates = (
        [candidate for candidate in build_candidates() if candidate.name in args.candidate.split(",")]
        if args.candidate
        else build_candidates()
    )
    summaries = []
    started = time.time()
    for candidate in candidates:
        print(f"==> {candidate.name}", flush=True)
        summary = evaluate_candidate(
            candidate, args.fold, args.fold_mode, angles, args.workers
        )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    result = {
        "mode": "validate",
        "fold": args.fold,
        "fold_mode": args.fold_mode,
        "angles": angles,
        "elapsed_seconds": time.time() - started,
        "summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file_pointer:
        json.dump(result, file_pointer, ensure_ascii=False, indent=2)
    return 0


def run_predict(args: argparse.Namespace) -> int:
    """为指定候选生成完整测试概率缓存。"""

    candidates = {
        candidate.name: candidate
        for candidate in build_candidates()
    }
    selected_names = args.candidate.split(",") if args.candidate else ["v5i_like"]
    selected = [candidates[name] for name in selected_names]
    (
        train_features,
        names,
        train_meta,
        train_cov,
        train_band,
        test_features,
        test_meta,
        test_cov,
        test_band,
        pairing,
    ) = load_data()
    test_meta["inferred_kind"] = np.where(
        test_meta["session_id"].eq("session1"),
        "unknown",
        np.where(test_meta["file_size_bytes"].gt(170000), "adversarial", "normal"),
    )
    twin_map = (
        pairing[["adv_sample_id", "clean_sample_id"]]
        .set_index("adv_sample_id")["clean_sample_id"]
        .to_dict()
    )

    for candidate in selected:
        probability = np.zeros((len(test_meta), 20), dtype=np.float32)
        for task in TASKS:
            train_mask = train_meta["task_label"].eq(task).to_numpy()
            test_mask = test_meta["task_label"].eq(task).to_numpy()
            normal_mask = test_meta.loc[test_mask, "inferred_kind"].eq("normal").to_numpy()
            normal_indices = np.where(test_mask)[0][normal_mask]
            labels = train_meta.loc[train_mask, "subject_id"].astype(int).to_numpy()
            train_design, target_design, train_labels, _ = build_design(
                task,
                np.arange(int(train_mask.sum())),
                np.arange(int(normal_mask.sum())),
                train_features[train_mask],
                names,
                train_cov[train_mask],
                train_band[train_mask],
                labels,
                candidate,
                test_features[normal_indices],
                test_cov[normal_indices],
                test_band[normal_indices],
            )
            raw = fit_predict_proba(
                train_design,
                target_design,
                train_labels,
                candidate,
                candidate.target_center,
            )
            probability[normal_indices] = calibrated_probability(
                raw, train_labels, "none"
            ).astype(np.float32)

        np.savez_compressed(
            REPORT_DIR / f"v6_probs_{candidate.name}.npz",
            probability=probability,
            sample_ids=test_meta["sample_id"].to_numpy(),
        )
        print(candidate.name, "完成", flush=True)
    return 0


def run_finalize(args: argparse.Namespace) -> int:
    """融合最终概率并生成官方 Excel。"""

    candidate_names = (
        args.candidate.split(",")
        if args.candidate
        else ["v5i_exact", "tan_all", "band_tan_robust", "lr_0.1"]
    )
    test_meta = pd.read_csv(FEATURE_DIR / "test_metadata.csv")
    test_meta["inferred_kind"] = np.where(
        test_meta["session_id"].eq("session1"),
        "unknown",
        np.where(test_meta["file_size_bytes"].gt(170000), "adversarial", "normal"),
    )
    weight_map = {}
    if args.weights:
        for item in args.weights.split(","):
            name, raw_weight = item.split(":", 1)
            weight_map[name] = float(raw_weight)

    probabilities: list[np.ndarray] = []
    weights: list[float] = []
    for name in candidate_names:
        if name == "v5i_exact":
            exact = np.zeros((len(test_meta), 20), dtype=np.float64)
            for task in TASKS:
                cache = np.load(REPORT_DIR / f"v5i_probs_{task}.npz")
                sample_ids = cache["sample_ids"]
                exact[np.asarray(sample_ids) - 1] = cache["probability"]
            probabilities.append(exact)
        else:
            cache = np.load(REPORT_DIR / f"v6_probs_{name}.npz")
            probabilities.append(cache["probability"])
        weights.append(weight_map.get(name, 1.0))
    probability = np.average(
        probabilities, axis=0, weights=weights
    ).astype(np.float64)
    train_meta = pd.read_csv(FEATURE_DIR / "train_metadata.csv")
    if args.calibration != "none":
        for task in TASKS:
            task_train_mask = train_meta["task_label"].eq(task).to_numpy()
            task_test_mask = test_meta["task_label"].eq(task).to_numpy()
            normal_mask = (
                test_meta.loc[task_test_mask, "inferred_kind"].eq("normal").to_numpy()
            )
            normal_indices = np.where(task_test_mask)[0][normal_mask]
            labels = train_meta.loc[
                task_train_mask, "subject_id"
            ].astype(int).to_numpy()
            if args.calibration == "train03":
                counts = np.asarray(
                    [np.sum(labels == label) for label in CLASSES], dtype=float
                )
                target_counts = counts / counts.sum() * len(normal_indices)
                calibrated = sinkhorn_calibrate(
                    probability[normal_indices], target_counts
                )
                probability[normal_indices] = (
                    0.7 * probability[normal_indices] + 0.3 * calibrated
                )
            else:
                calibrated = sinkhorn_calibrate(
                    probability[normal_indices],
                    np.full(20, len(normal_indices) / 20.0),
                )
                probability[normal_indices] = (
                    0.7 * probability[normal_indices] + 0.3 * calibrated
                )
    normal_prediction = CLASSES[probability.argmax(axis=1)]
    confidence = probability.max(axis=1)
    pairing = pd.read_csv(PAIRING_PATH)
    twin_map = (
        pairing[["adv_sample_id", "clean_sample_id"]]
        .set_index("adv_sample_id")["clean_sample_id"]
        .to_dict()
    )
    normal_by_sample = pd.Series(normal_prediction, index=test_meta["sample_id"])
    predicted_label = normal_prediction.copy()
    adversarial = test_meta["inferred_kind"].eq("adversarial").to_numpy()
    mapped_labels = (
        test_meta.loc[adversarial, "sample_id"]
        .map(twin_map)
        .map(normal_by_sample)
        .to_numpy()
    )
    predicted_label[adversarial] = mapped_labels
    clean_ids = test_meta.loc[adversarial, "sample_id"].map(twin_map).to_numpy()
    sample_index = pd.Series(
        np.arange(len(test_meta)), index=test_meta["sample_id"]
    )
    twin_confidence = confidence[sample_index.reindex(clean_ids).to_numpy()]
    adversarial_indices = np.where(adversarial)[0]
    predicted_label[adversarial_indices[twin_confidence < args.reject_threshold]] = 0
    predicted_label[test_meta["inferred_kind"].eq("unknown").to_numpy()] = 0

    template = pd.read_csv(
        ROOT / "Brainprint_Competition" / "test.csv", dtype={"Sample": str}
    )
    mapping = pd.Series(predicted_label, index=test_meta["sample_id"])
    template["Label"] = (
        template["Sample"].str.extract(r"(\d+)")[0].astype(int).map(mapping).astype(int)
    )
    final_path = SUBMISSION_DIR / "test_submission_v6_final.xlsx"
    mail_path = ROOT / "队名-第三次.xlsx"
    template.to_excel(final_path, index=False, sheet_name="Sheet1")
    template.to_excel(mail_path, index=False, sheet_name="Sheet1")
    template.to_csv(
        SUBMISSION_DIR / "test_submission_v6_final.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result = {
        "candidates": candidate_names,
        "weights": weights,
        "rows": len(template),
        "label_counts": template["Label"].value_counts().sort_index().to_dict(),
        "unknown_zero": bool(
            (template.loc[test_meta["inferred_kind"].eq("unknown").to_numpy(), "Label"] == 0).all()
        ),
        "adversarial_rejected": int(
            (template.loc[adversarial, "Label"] == 0).sum()
        ),
        "adversarial_total": int(adversarial.sum()),
        "unknown_total": int(test_meta["inferred_kind"].eq("unknown").sum()),
        "calibration": args.calibration,
        "reject_threshold": args.reject_threshold,
        "final_xlsx": str(final_path),
        "mail_xlsx": str(mail_path),
        "sha256": hashlib.sha256(template["Label"].astype(str).to_csv(index=False).encode()).hexdigest(),
    }
    with (REPORT_DIR / "v6_final_validation.json").open("w", encoding="utf-8") as file_pointer:
        json.dump(result, file_pointer, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    """运行脚本入口。"""

    args = parse_args()
    if args.mode == "validate":
        return run_validate(args)
    if args.mode == "predict":
        return run_predict(args)
    return run_finalize(args)


if __name__ == "__main__":
    raise SystemExit(main())
