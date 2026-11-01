"""比赛 EEG 的频域、空间相关与质量特征。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch


BANDS = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 40.0),
)


@dataclass(frozen=True)
class FeatureOutput:
    """保存一个 trial 的特征、名称和质量指标。"""

    values: np.ndarray
    names: tuple[str, ...]
    quality: dict[str, float]


def _bandpass(data: np.ndarray, fs: int, low: float = 1.0, high: float = 40.0) -> np.ndarray:
    """对 time × channel 数据进行零相位带通滤波。"""

    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=0)


def _spectral_entropy(power: np.ndarray) -> float:
    """计算归一化 PSD 的 Shannon 熵。"""

    probability = power / max(float(power.sum()), 1e-12)
    return float(-(probability * np.log2(probability + 1e-12)).sum())


def extract_features(data: np.ndarray, fs: int) -> FeatureOutput:
    """从 4 秒 trial 中提取可解释的稳健特征。"""

    signal = np.asarray(data, dtype=np.float64)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    signal = signal - signal.mean(axis=0, keepdims=True)
    scale = signal.std(axis=0, keepdims=True)
    signal = signal / np.maximum(scale, 1e-8)
    filtered = _bandpass(signal, fs=fs, low=1.0, high=40.0)

    frequencies, power = welch(
        filtered,
        fs=fs,
        window="hann",
        nperseg=min(300, filtered.shape[0]),
        noverlap=min(150, filtered.shape[0] // 2),
        axis=0,
        detrend="constant",
        scaling="density",
    )

    values: list[np.ndarray] = []
    names: list[str] = []
    relative_by_channel: list[np.ndarray] = []

    usable = (frequencies >= 1.0) & (frequencies <= 40.0)
    total_power = np.trapezoid(power[usable], frequencies[usable], axis=0)
    values.append(np.log10(total_power + 1e-12))
    names.extend(f"log_total_power_{index:02d}" for index in range(filtered.shape[1]))

    for band_name, low, high in BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        band_power = np.trapezoid(power[mask], frequencies[mask], axis=0)
        values.append(np.log10(band_power + 1e-12))
        names.extend(f"log_{band_name}_{index:02d}" for index in range(filtered.shape[1]))
        relative_by_channel.append(band_power / np.maximum(total_power, 1e-12))

    relative = np.stack(relative_by_channel, axis=1)
    values.append(relative.reshape(-1))
    names.extend(
        f"relative_{band_name}_{channel:02d}"
        for channel in range(filtered.shape[1])
        for band_name, _, _ in BANDS
    )

    entropy = np.array([_spectral_entropy(power[usable, channel]) for channel in range(filtered.shape[1])])
    values.append(entropy)
    names.extend(f"spectral_entropy_{index:02d}" for index in range(filtered.shape[1]))

    correlation = np.corrcoef(filtered, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    upper_index = np.triu_indices(filtered.shape[1], k=1)
    values.append(correlation[upper_index])
    names.extend(f"corr_{left:02d}_{right:02d}" for left, right in zip(*upper_index))

    derivative_rms = np.sqrt(np.mean(np.diff(filtered, axis=0) ** 2, axis=0))
    values.append(np.log10(derivative_rms + 1e-12))
    names.extend(f"log_derivative_rms_{index:02d}" for index in range(filtered.shape[1]))

    # 质量指标不直接参与身份分类，但用于识别高风险测试文件。
    full_freq, full_power = welch(signal, fs=fs, nperseg=min(300, signal.shape[0]), axis=0)
    low_mask = (full_freq >= 1) & (full_freq <= 40)
    high_mask = (full_freq > 40) & (full_freq <= min(120, fs / 2))
    low_energy = np.trapezoid(full_power[low_mask], full_freq[low_mask], axis=0).sum()
    high_energy = np.trapezoid(full_power[high_mask], full_freq[high_mask], axis=0).sum()
    quality = {
        "high_frequency_ratio": float(high_energy / max(low_energy, 1e-12)),
        "flat_channels": float(np.sum(scale.ravel() < 1e-8)),
        "p99_abs": float(np.percentile(np.abs(signal), 99)),
        "channel_std_median": float(np.median(scale.ravel())),
    }

    return FeatureOutput(
        values=np.concatenate(values).astype(np.float32),
        names=tuple(names),
        quality=quality,
    )
