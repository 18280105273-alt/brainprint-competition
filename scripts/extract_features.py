#!/usr/bin/env python3
"""并行提取训练集和测试集的 EEG 特征。"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brainprint_comp.data import load_trial, sample_number, subject_number, task_number
from brainprint_comp.features import extract_features


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-root", type=Path, default=ROOT / "Brainprint_Competition")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "features")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


def extract_one(path: Path, split: str) -> tuple[dict[str, Any], np.ndarray, tuple[str, ...]]:
    """提取一个文件的特征及元数据。"""

    path = path.resolve()
    record = load_trial(path)
    output = extract_features(record.data, record.fs)
    metadata = {
        "split": split,
        "file_path": str(path.relative_to(ROOT)),
        "sample_id": sample_number(path),
        "subject_id": subject_number(path),
        "task_dir": task_number(path),
        "session_id": record.session_id,
        "task_label": record.task_label,
        "file_size_bytes": path.stat().st_size,
        **output.quality,
    }
    return metadata, output.values, output.names


def extract_many(files: list[Path], split: str, workers: int) -> tuple[pd.DataFrame, np.ndarray, tuple[str, ...]]:
    """并行提取多个文件，并按原文件顺序排列。"""

    started = time.time()
    results: list[tuple[int, dict[str, Any], np.ndarray, tuple[str, ...]]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_one, path, split): (index, path) for index, path in enumerate(files)}
        for done, future in enumerate(as_completed(futures), start=1):
            index, _path = futures[future]
            metadata, values, names = future.result()
            results.append((index, metadata, values, names))
            if done % 1000 == 0 or done == len(files):
                print(f"[{split}] {done}/{len(files)}，{time.time() - started:.1f}s", flush=True)

    results.sort(key=lambda item: item[0])
    metadata = pd.DataFrame([item[1] for item in results])
    matrix = np.stack([item[2] for item in results]).astype(np.float32)
    names = results[0][3]
    return metadata, matrix, names


def main() -> None:
    """执行训练集和测试集特征提取。"""

    args = parse_args()
    root = args.competition_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_files = sorted((root / "Train").rglob("*.mat"))
    test_files = sorted((root / "Test").glob("*.mat"), key=lambda path: int(path.stem.split("_")[-1]))

    train_meta, train_x, names = extract_many(train_files, "train", args.workers)
    test_meta, test_x, test_names = extract_many(test_files, "test", args.workers)
    if names != test_names:
        raise RuntimeError("训练集和测试集特征列不一致")

    np.savez_compressed(
        args.output_dir / "train_features.npz",
        x=train_x,
        names=np.asarray(names),
    )
    np.savez_compressed(
        args.output_dir / "test_features.npz",
        x=test_x,
        names=np.asarray(names),
    )
    train_meta.to_csv(args.output_dir / "train_metadata.csv", index=False, encoding="utf-8-sig")
    test_meta.to_csv(args.output_dir / "test_metadata.csv", index=False, encoding="utf-8-sig")
    print(f"训练特征 {train_x.shape}，测试特征 {test_x.shape}，特征列 {len(names)}")


if __name__ == "__main__":
    main()
