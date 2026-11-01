"""统一运行入口。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

COMMANDS = {
    "audit": ("scripts/audit_competition.py", ["--workers", "8"]),
    "features": ("scripts/extract_features.py", ["--workers", "8"]),
    "covariances": ("scripts/extract_covariances.py", ["--workers", "8"]),
    "train": ("scripts/train_final_tangent.py", []),
    "ensemble": ("scripts/search_ensemble.py", []),
    "predict": ("scripts/generate_v4_ensemble.py", []),
}

PREREQUISITES = (
    "outputs/features/train_features.npz",
    "outputs/features/test_features.npz",
    "outputs/features/train_covariances.npy",
    "outputs/features/test_covariances.npy",
    "outputs/reports/final_tangent_metrics.csv",
    "outputs/reports/ensemble_search.csv",
)


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(prog="brainprint-competition")
    parser.add_argument(
        "command",
        nargs="?",
        choices=[*sorted(COMMANDS), "pipeline"],
        help="要执行的流程",
    )
    parser.add_argument("--list", action="store_true", help="列出可执行流程")
    return parser


def main(argv: list[str] | None = None) -> int:
    """执行指定流程。"""

    arguments = build_parser().parse_args(argv)
    if arguments.list or arguments.command is None:
        print("可用流程：")
        for name in [*sorted(COMMANDS), "pipeline"]:
            print(f"- {name}")
        return 0
    if arguments.command == "pipeline":
        order = ["audit", "features", "covariances", "train", "ensemble", "predict"]
        for name in order:
            print(f"==> {name}", flush=True)
            code = run_command(name)
            if code != 0:
                return code
        return 0
    if arguments.command == "predict":
        missing = [path for path in PREREQUISITES if not (ROOT / path).exists()]
        if missing:
            print("检测到预测前置文件缺失，自动运行完整流水线。", flush=True)
            for name in ["features", "covariances", "train", "ensemble"]:
                print(f"==> {name}", flush=True)
                code = run_command(name)
                if code != 0:
                    return code
    return run_command(arguments.command)


def run_command(name: str) -> int:
    """运行单个命令并返回退出码。"""

    script, extra = COMMANDS[name]
    command = [sys.executable, str(ROOT / script), *extra]
    return subprocess.call(command, cwd=ROOT)
