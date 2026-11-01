#!/usr/bin/env python3
"""提取比赛 trial 的空间协方差矩阵，供黎曼切空间模型使用。"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brainprint_comp.data import load_trial, portable_path, sample_number, subject_number, task_number


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-root", type=Path, default=ROOT / "Brainprint_Competition")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "features")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


def covariance_one(path: Path, split: str) -> tuple[str, np.ndarray, dict]:
    """读取一条 trial 并计算正则化协方差矩阵。"""

    path = path.resolve()
    record = load_trial(path)
    signal = record.data.astype(np.float64)
    sos = butter(4, [1.0, 40.0], btype="bandpass", fs=record.fs, output="sos")
    signal = sosfiltfilt(sos, signal, axis=0)
    signal = (signal - signal.mean(axis=0, keepdims=True)) / np.maximum(
        signal.std(axis=0, keepdims=True),
        1e-8,
    )
    covariance = np.cov(signal, rowvar=False).astype(np.float32)
    covariance += np.eye(covariance.shape[0], dtype=np.float32) * 1e-3
    return (
        str(path),
        covariance,
        {
            "split": split,
            "file_path": portable_path(path, ROOT),
            "sample_id": sample_number(path),
            "subject_id": subject_number(path),
            "task_dir": task_number(path),
            "session_id": record.session_id,
            "task_label": record.task_label,
            "file_size_bytes": path.stat().st_size,
        },
    )


def extract_many(files: list[Path], split: str, workers: int) -> tuple[pd.DataFrame, np.ndarray]:
    """并行提取协方差矩阵并按文件名排序。"""

    started = time.time()
    results: list[tuple[str, np.ndarray, dict]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(covariance_one, path, split): path for path in files}
        for done, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if done % 1000 == 0 or done == len(files):
                print(f"[{split}] {done}/{len(files)}，{time.time() - started:.1f}s", flush=True)
    if split == "test":
        results.sort(key=lambda item: int(Path(item[0]).stem.split("_")[-1]))
    else:
        results.sort(key=lambda item: item[0])
    return (
        pd.DataFrame([item[2] for item in results]),
        np.stack([item[1] for item in results]),
    )


def main() -> None:
    """提取训练集和测试集协方差矩阵。"""

    args = parse_args()
    root = args.competition_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_files = sorted((root / "Train").rglob("*.mat"))
    test_files = sorted((root / "Test").glob("*.mat"), key=lambda path: int(path.stem.split("_")[-1]))
    train_meta, train_cov = extract_many(train_files, "train", args.workers)
    test_meta, test_cov = extract_many(test_files, "test", args.workers)
    np.save(args.output_dir / "train_covariances.npy", train_cov)
    np.save(args.output_dir / "test_covariances.npy", test_cov)
    train_meta.to_csv(args.output_dir / "train_covariance_metadata.csv", index=False, encoding="utf-8-sig")
    test_meta.to_csv(args.output_dir / "test_covariance_metadata.csv", index=False, encoding="utf-8-sig")
    print(f"训练协方差 {train_cov.shape}，测试协方差 {test_cov.shape}")


if __name__ == "__main__":
    main()
