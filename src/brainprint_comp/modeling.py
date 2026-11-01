"""经典机器学习模型、特征选择和测试预测。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROBUST_PREFIXES = ("relative_", "spectral_entropy_", "corr_")


@dataclass
class TaskModel:
    """保存一个任务模型及其训练信息。"""

    task: str
    estimator: Pipeline
    feature_names: np.ndarray
    classes: np.ndarray


def select_features(names: Iterable[str], mode: str) -> np.ndarray:
    """选择全部特征或抗高频扰动特征。"""

    names = np.asarray(list(names)).astype(str)
    if mode == "all":
        return np.ones(len(names), dtype=bool)
    if mode == "robust":
        return np.asarray([name.startswith(ROBUST_PREFIXES) for name in names], dtype=bool)
    raise ValueError(f"未知特征模式: {mode}")


def build_logistic_regression() -> Pipeline:
    """构造标准化加多项逻辑回归模型。"""

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=500,
                    solver="lbfgs",
                    random_state=2026,
                ),
            ),
        ]
    )


def fit_task_model(
    task: str,
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: np.ndarray,
    mode: str,
) -> TaskModel:
    """按任务训练一个身份分类器。"""

    mask = select_features(feature_names, mode)
    estimator = build_logistic_regression()
    estimator.fit(features[:, mask], labels)
    return TaskModel(
        task=task,
        estimator=estimator,
        feature_names=feature_names[mask],
        classes=np.asarray(estimator.classes_),
    )


def predict_proba(model: TaskModel, features: np.ndarray, feature_names: np.ndarray) -> np.ndarray:
    """使用指定特征列进行概率预测。"""

    mask = np.isin(feature_names, model.feature_names)
    return model.estimator.predict_proba(features[:, mask])
