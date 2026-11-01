#!/usr/bin/env python3
"""全量审计脑纹比赛训练集、测试集和提交模板。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brainprint_comp.data import load_trial, portable_path, sample_number, subject_number, task_number


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-root", type=Path, default=ROOT / "Brainprint_Competition")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "audit")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


def audit_one(path: Path, split: str) -> dict[str, Any]:
    """读取一个 trial 并提取质量与元数据指标。"""

    record = load_trial(path)
    data = record.data
    channel_std = np.nanstd(data, axis=0)
    abs_values = np.abs(data)
    row = {
        "split": split,
        "file_path": portable_path(path, ROOT),
        "file_name": path.name,
        "file_size_bytes": path.stat().st_size,
        "subject_id": subject_number(path),
        "task_dir": task_number(path),
        "sample_id": sample_number(path),
        "session_id": record.session_id,
        "task_label": record.task_label,
        "fs": record.fs,
        "time_points": int(data.shape[0]),
        "channels": int(data.shape[1]),
        "channel_names": "|".join(record.channel_names),
        "unit": record.unit,
        "finite_fraction": float(np.isfinite(data).mean()),
        "flat_channels": int(np.sum(channel_std < 1e-8)),
        "channel_std_median": float(np.median(channel_std)),
        "channel_std_min": float(np.min(channel_std)),
        "channel_std_max": float(np.max(channel_std)),
        "abs_p99": float(np.percentile(abs_values, 99)),
        "abs_max": float(np.max(abs_values)),
        "ptp_median": float(np.median(np.ptp(data, axis=0))),
        "error": "",
    }
    return row


def audit_many(files: list[Path], split: str, workers: int) -> pd.DataFrame:
    """并行审计一组文件，并持续输出进度。"""

    rows: list[dict[str, Any]] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(audit_one, path, split): path for path in files}
        for done, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append(
                    {
                        "split": split,
                        "file_path": portable_path(path, ROOT),
                        "file_name": path.name,
                        "file_size_bytes": path.stat().st_size if path.exists() else 0,
                        "subject_id": subject_number(path),
                        "task_dir": task_number(path),
                        "sample_id": sample_number(path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            if done % 1000 == 0 or done == len(files):
                elapsed = time.time() - started
                print(f"[{split}] {done}/{len(files)} files, {elapsed:.1f}s", flush=True)
    return pd.DataFrame(rows)


def read_submission_template(path: Path) -> pd.DataFrame:
    """读取官方提交模板并校验基础列。"""

    frame = pd.read_csv(path, dtype=str)
    required = {"Sample", "Label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"test.csv 缺少列: {sorted(missing)}")
    return frame


def summarize(train: pd.DataFrame, test: pd.DataFrame, template: pd.DataFrame) -> dict[str, Any]:
    """生成可写入 JSON 的审计摘要。"""

    def counts(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
        if frame.empty:
            return {}
        return {
            " | ".join(map(str, key)) if isinstance(key, tuple) else str(key): int(value)
            for key, value in frame.groupby(columns, dropna=False).size().items()
        }

    template_samples = template["Sample"].astype(str).tolist()
    test_samples = [f"test_{int(value):03d}" for value in test["sample_id"].dropna()]
    missing_files = sorted(set(template_samples) - set(test_samples))
    extra_files = sorted(set(test_samples) - set(template_samples))

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "counts": {
            "train_files": int(len(train)),
            "test_files": int(len(test)),
            "submission_rows": int(len(template)),
        },
        "train_subjects": int(train["subject_id"].nunique(dropna=True)),
        "train_subject_rows": {
            str(int(key)): int(value)
            for key, value in train.groupby("subject_id", dropna=True).size().items()
        },
        "train_task_session": counts(train, ["task_label", "session_id"]),
        "test_task_session": counts(test, ["task_label", "session_id"]),
        "train_task_subject_session": counts(train, ["task_label", "session_id"]),
        "shapes": {
            "train": sorted({(int(r.time_points), int(r.channels)) for r in train.itertuples()}),
            "test": sorted({(int(r.time_points), int(r.channels)) for r in test.itertuples()}),
        },
        "sampling_rates": {
            "train": sorted(int(value) for value in train["fs"].dropna().unique()),
            "test": sorted(int(value) for value in test["fs"].dropna().unique()),
        },
        "channel_sets": {
            "train": sorted(train["channel_names"].dropna().unique().tolist()),
            "test": sorted(test["channel_names"].dropna().unique().tolist()),
        },
        "errors": {
            "train": int((train.get("error", "") != "").sum()),
            "test": int((test.get("error", "") != "").sum()),
        },
        "quality": {
            "train_abs_p99_quantiles": train["abs_p99"].quantile([0.5, 0.9, 0.95, 0.99]).round(4).to_dict()
            if "abs_p99" in train
            else {},
            "test_abs_p99_quantiles": test["abs_p99"].quantile([0.5, 0.9, 0.95, 0.99]).round(4).to_dict()
            if "abs_p99" in test
            else {},
            "train_rows_with_flat_channels": int((train.get("flat_channels", 0) > 0).sum()),
            "test_rows_with_flat_channels": int((test.get("flat_channels", 0) > 0).sum()),
        },
        "submission_alignment": {
            "missing_files": missing_files[:20],
            "extra_files": extra_files[:20],
            "missing_count": len(missing_files),
            "extra_count": len(extra_files),
            "label_nonempty_count": int(template["Label"].fillna("").astype(str).str.strip().ne("").sum()),
        },
    }


def main() -> None:
    """执行全量审计。"""

    args = parse_args()
    competition_root = args.competition_root.resolve()
    train_root = competition_root / "Train"
    test_root = competition_root / "Test"
    template_path = competition_root / "test.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_files = sorted(train_root.rglob("*.mat"))
    test_files = sorted(test_root.glob("*.mat"))
    print(f"训练集 {len(train_files)}，测试集 {len(test_files)}，workers={args.workers}")

    train_audit = audit_many(train_files, "train", args.workers)
    test_audit = audit_many(test_files, "test", args.workers)
    template = read_submission_template(template_path)

    train_audit.sort_values("file_path").to_csv(args.output_dir / "train_audit.csv", index=False, encoding="utf-8-sig")
    test_audit.sort_values("sample_id").to_csv(args.output_dir / "test_audit.csv", index=False, encoding="utf-8-sig")

    summary = summarize(train_audit, test_audit, template)
    with (args.output_dir / "audit_summary.json").open("w", encoding="utf-8") as file_pointer:
        json.dump(summary, file_pointer, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
