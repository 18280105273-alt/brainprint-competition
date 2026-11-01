"""赛事综合评分公式。"""

from __future__ import annotations


def official_score(clean_accuracy: float, adversarial_score: float, unknown_rejection: float) -> float:
    """计算官方加权几何平均分。"""

    if clean_accuracy <= 0 or adversarial_score <= 0 or unknown_rejection <= 0:
        return 0.0
    return (
        100.0
        * clean_accuracy**0.40
        * adversarial_score**0.35
        * unknown_rejection**0.25
    )
