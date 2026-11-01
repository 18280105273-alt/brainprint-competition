"""比赛 MAT 文件的读取与元数据解析。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio


@dataclass(frozen=True)
class TrialRecord:
    """保存单个 trial 的路径、信号和元数据。"""

    path: Path
    data: np.ndarray
    fs: int
    channel_names: tuple[str, ...]
    session_id: str
    task_label: str
    unit: str


def _mat_string(value: Any) -> str:
    """把 MATLAB 字符数组或 cell 字符转换为普通字符串。"""

    array = np.asarray(value)
    if array.size == 0:
        return ""
    if array.dtype.kind in {"U", "S"}:
        return "".join(str(item) for item in array.ravel())
    if array.dtype == object:
        return "".join(_mat_string(item) for item in array.ravel())
    return str(array.ravel()[0])


def _channel_names(value: Any) -> tuple[str, ...]:
    """解析 MAT 中的通道名称 cell 数组。"""

    array = np.asarray(value)
    if array.dtype != object:
        return tuple(str(item) for item in array.ravel())
    return tuple(_mat_string(item) for item in array.ravel())


def load_trial(path: Path) -> TrialRecord:
    """读取一条比赛 trial，并统一为 time × channel 的 float32 数组。"""

    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    data = np.asarray(mat["data"], dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    fs = int(np.asarray(mat["fs"]).ravel()[0])
    return TrialRecord(
        path=path,
        data=data,
        fs=fs,
        channel_names=_channel_names(mat["channel_names"]),
        session_id=_mat_string(mat.get("session_id", "")),
        task_label=_mat_string(mat.get("task_label", "")),
        unit=_mat_string(mat.get("unit", "")),
    )


def subject_number(path: Path) -> int | None:
    """从训练路径中的 Subject_XX 提取身份编号。"""

    for part in path.parts:
        if part.startswith("Subject_") and part[8:].isdigit():
            return int(part[8:])
    return None


def task_number(path: Path) -> int | None:
    """从训练路径中的 TaskN 提取任务编号。"""

    for part in path.parts:
        if part.startswith("Task") and part[4:].isdigit():
            return int(part[4:])
    return None


def sample_number(path: Path) -> int | None:
    """从 test_XXXXX.mat 提取测试样本编号。"""

    stem = path.stem
    if stem.startswith("test_") and stem[5:].isdigit():
        return int(stem[5:])
    return None
