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
    "predict": ("scripts/generate_v3_tlcenter.py", []),
}


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(prog="brainprint-competition")
    parser.add_argument(
        "command",
        nargs="?",
        choices=sorted(COMMANDS),
        help="要执行的流程",
    )
    parser.add_argument("--list", action="store_true", help="列出可执行流程")
    return parser


def main(argv: list[str] | None = None) -> int:
    """执行指定流程。"""

    arguments = build_parser().parse_args(argv)
    if arguments.list or arguments.command is None:
        print("可用流程：")
        for name in sorted(COMMANDS):
            print(f"- {name}")
        return 0
    script, extra = COMMANDS[arguments.command]
    command = [sys.executable, str(ROOT / script), *extra]
    return subprocess.call(command, cwd=ROOT)
