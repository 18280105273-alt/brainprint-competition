#!/usr/bin/env python3
"""生成 v5 确定性修复候选，并比较不同强度的先验软校准。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyriemann.geometry.mean import mean_covariance
from pyriemann.geometry.tangentspace import tangent_space
from pyriemann.transfer import TLCenter, encode_domains
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.train_baseline import sinkhorn_calibrate


TASKS = ("EC", "EO", "Imagery", "SSVEP")
FEATURE_DIR = ROOT / "outputs" / "features"
REPORT_DIR = ROOT / "outputs" / "reports"
SUBMISSION_DIR = ROOT / "outputs" / "submissions"
PAIRING_PATH = next(
    path
    for path in (
        ROOT / "outputs" / "reports" / "adv_clean_pairs.csv",
        Path(
            "/Users/lizhongxing/Desktop/动脉探测器/brainprint_analysis/"
            "agents/data_forensics/metrics/adv_clean_pairs.csv"
        ),
    )
    if path.exists()
)
WEIGHTS = {
    "EC": {"tan_rob": 0.7, "tan_all": 0.2, "tan_relcorr": 0.1},
    "EO": {"tan_all": 1.0},
    "Imagery": {"tan_all": 1.0},
    "SSVEP": {"tan_rob": 0.6, "tan_all": 0.4},
}


def classifier() -> LogisticRegression:
    """固定分类器。"""

    return LogisticRegression(C=1.0, max_iter=600, solver="lbfgs", random_state=2026)


def align_tangent(source_cov, target_cov, source_labels):
    """执行目标协方差中心对齐并返回源/目标切空间。"""

    all_cov = np.concatenate([source_cov, target_cov])
    domains = np.concatenate(
        [
            np.zeros(len(source_cov), dtype=int),
            np.ones(len(target_cov), dtype=int),
        ]
    )
    labels = np.concatenate([source_labels, np.zeros(len(target_cov), dtype=int)])
    encoded_x, encoded_y = encode_domains(all_cov, labels, domains)
    aligned = TLCenter(target_domain=1, metric="riemann").fit_transform(
        encoded_x,
        encoded_y,
    )
    reference = mean_covariance(aligned[: len(source_cov)], metric="riemann")
    tangent = tangent_space(aligned, reference, metric="riemann")
    return tangent[: len(source_cov)], tangent[len(source_cov) :]


def task_probabilities(
    task,
    train_features,
    train_cov,
    train_labels,
    test_features,
    test_cov,
    normal_mask,
    feature_names,
    robust_mask,
    relcorr_mask,
    use_tlcenter,
):
    """计算普通测试样本的任务概率。"""

    design_sets = {
        "tan_rob": (train_features[:, robust_mask], test_features[normal_mask][:, robust_mask]),
        "tan_all": (train_features, test_features[normal_mask]),
        "tan_relcorr": (train_features[:, relcorr_mask], test_features[normal_mask][:, relcorr_mask]),
    }
    if use_tlcenter:
        source_tangent, target_tangent = align_tangent(
            train_cov,
            test_cov[normal_mask],
            train_labels,
        )
    else:
        reference = mean_covariance(train_cov, metric="riemann")
        all_tangent = tangent_space(
            np.concatenate([train_cov, test_cov[normal_mask]]),
            reference,
            metric="riemann",
        )
        source_tangent = all_tangent[: len(train_cov)]
        target_tangent = all_tangent[len(train_cov) :]

    probability = np.zeros((int(normal_mask.sum()), 20), dtype=np.float64)
    for design_name, (source_feature, target_feature) in design_sets.items():
        source_design = np.hstack([source_tangent, source_feature])
        target_design = np.hstack([target_tangent, target_feature])
        scaler = StandardScaler().fit(source_design)
        model = classifier().fit(scaler.transform(source_design), train_labels)
        probability += WEIGHTS[task].get(design_name, 0.0) * model.predict_proba(
            scaler.transform(target_design)
        )
    probability /= sum(WEIGHTS[task].values())
    return probability


def save_submission(prediction, name):
    """生成 Excel 和 CSV。"""

    template = pd.read_csv(
        ROOT / "Brainprint_Competition" / "test.csv",
        dtype={"Sample": str},
    )
    mapping = prediction.set_index("sample_id")["predicted_label"]
    template["Label"] = (
        template["Sample"].str.extract(r"(\d+)")[0].astype(int).map(mapping).astype(int)
    )
    template.to_excel(
        SUBMISSION_DIR / f"test_submission_{name}.xlsx",
        index=False,
        sheet_name="Sheet1",
    )
    template.to_csv(
        SUBMISSION_DIR / f"test_submission_{name}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return template


def spectral_view_probabilities(
    train_features,
    test_features,
    feature_names,
):
    """独立纯频谱视图，用于两视图一致率代理。"""

    mask = np.asarray(
        [name.startswith(("relative_", "corr_")) for name in feature_names]
    )
    outputs = np.zeros((len(test_features), 20), dtype=np.float64)
    for task in TASKS:
        train_mask = train_meta["task_label"].eq(task).to_numpy()
        test_mask = test_meta["task_label"].eq(task).to_numpy()
        task_labels = train_meta.loc[train_mask, "subject_id"].astype(int).to_numpy()
        scaler = StandardScaler().fit(train_features[train_mask][:, mask])
        model = classifier().fit(
            scaler.transform(train_features[train_mask][:, mask]),
            task_labels,
        )
        outputs[test_mask] = model.predict_proba(
            scaler.transform(test_features[test_mask][:, mask])
        )
    return np.arange(1, 21)[outputs.argmax(axis=1)]


def main():
    """生成 v5 候选、Excel 文件和代理指标。"""

    started = time.time()
    train_npz = np.load(FEATURE_DIR / "train_features.npz", allow_pickle=True)
    test_npz = np.load(FEATURE_DIR / "test_features.npz", allow_pickle=True)
    train_features = train_npz["x"]
    test_features = test_npz["x"]
    feature_names = np.asarray(train_npz["names"]).astype(str)
    robust_mask = np.asarray(
        [name.startswith(("relative_", "spectral_entropy_", "corr_")) for name in feature_names]
    )
    relcorr_mask = np.asarray(
        [name.startswith(("relative_", "corr_")) for name in feature_names]
    )
    global train_meta, test_meta
    train_meta = pd.read_csv(FEATURE_DIR / "train_metadata.csv")
    test_meta = pd.read_csv(FEATURE_DIR / "test_metadata.csv")
    train_cov = np.load(FEATURE_DIR / "train_covariances.npy")
    test_cov = np.load(FEATURE_DIR / "test_covariances.npy")
    pairing = pd.read_csv(PAIRING_PATH)

    test_meta["inferred_kind"] = np.where(
        test_meta["session_id"].eq("session1"),
        "unknown",
        np.where(test_meta["file_size_bytes"] > 170000, "adversarial", "normal"),
    )
    train_labels_by_task = {
        task: train_meta.loc[
            train_meta["task_label"].eq(task), "subject_id"
        ].astype(int).to_numpy()
        for task in TASKS
    }

    candidates = {}
    for name, use_tlcenter, calibration_alpha in [
        ("v5a_raw", False, 0.0),
        ("v5a_tlc", True, 0.0),
        ("v5b_fixed_raw", False, 1.0),
        ("v5b_fixed_tlc", True, 1.0),
        ("v5c_soft03", True, 0.3),
    ]:
        normal_probability = np.zeros((len(test_meta), 20), dtype=np.float32)
        for task in TASKS:
            train_mask = train_meta["task_label"].eq(task).to_numpy()
            test_mask = test_meta["task_label"].eq(task).to_numpy()
            normal_mask = test_meta.loc[test_mask, "inferred_kind"].eq("normal").to_numpy()
            normal_indices = np.where(test_mask)[0][normal_mask]
            task_probability = task_probabilities(
                task,
                train_features[train_mask],
                train_cov[train_mask],
                train_labels_by_task[task],
                test_features[test_mask],
                test_cov[test_mask],
                normal_mask,
                feature_names,
                robust_mask,
                relcorr_mask,
                use_tlcenter,
            ).astype(np.float64)
            if calibration_alpha > 0.0:
                task_counts = np.asarray(
                    [
                        np.sum(train_labels_by_task[task] == label)
                        for label in range(1, 21)
                    ],
                    dtype=np.float64,
                )
                calibrated_probability = sinkhorn_calibrate(
                    task_probability,
                    task_counts / task_counts.sum() * len(normal_indices),
                )
                task_probability = (
                    (1.0 - calibration_alpha) * task_probability
                    + calibration_alpha * calibrated_probability
                )
            normal_probability[normal_indices] = task_probability.astype(np.float32)

        normal_prediction = np.arange(1, 21)[normal_probability.argmax(axis=1)]
        predicted_label = normal_prediction.copy()
        confidence = normal_probability.max(axis=1)
        normal_by_sample = pd.DataFrame(
            {
                "sample_id": test_meta["sample_id"],
                "normal_prediction": normal_prediction,
            }
        ).set_index("sample_id")["normal_prediction"]

        twin_map = (
            pairing[["adv_sample_id", "clean_sample_id"]]
            .set_index("adv_sample_id")["clean_sample_id"]
            .to_dict()
        )
        mapped_labels = test_meta["sample_id"].map(twin_map).map(normal_by_sample)
        adversarial_mask = test_meta["inferred_kind"].eq("adversarial").to_numpy()
        predicted_label[adversarial_mask] = mapped_labels[adversarial_mask].to_numpy()
        sample_index = pd.Series(
            np.arange(len(test_meta)),
            index=test_meta["sample_id"],
        )
        clean_ids = test_meta.loc[adversarial_mask, "sample_id"].map(twin_map).astype(int)
        confidence[adversarial_mask] = confidence[sample_index.reindex(clean_ids).to_numpy()]
        unknown_mask = test_meta["inferred_kind"].eq("unknown").to_numpy()
        predicted_label[unknown_mask] = 0
        confidence[unknown_mask] = 0.0

        prediction = test_meta[
            ["sample_id", "task_label", "session_id", "file_size_bytes", "inferred_kind"]
        ].copy()
        prediction["predicted_label"] = predicted_label
        prediction["confidence"] = confidence
        prediction.to_csv(
            REPORT_DIR / f"final_predictions_{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        save_submission(prediction, name)
        candidates[name] = prediction

    spectral_prediction = spectral_view_probabilities(
        train_features,
        test_features,
        feature_names,
    )
    summary = {
        "elapsed_seconds": round(time.time() - started, 3),
        "pairing": {
            "adv_rows": int(len(pairing)),
            "unique_clean": int(pairing["clean_sample_id"].nunique()),
            "min_similarity": float(pairing["sim1"].min()),
            "median_similarity": float(pairing["sim1"].median()),
            "all_normal_pool": bool((pairing["clean_kind"] == "normal").all()),
        },
    }
    for name, prediction in candidates.items():
        normal = prediction[prediction["inferred_kind"] == "normal"]
        adversarial = prediction[prediction["inferred_kind"] == "adversarial"]
        summary[name] = {
            "label_counts": prediction["predicted_label"].value_counts().sort_index().astype(int).to_dict(),
            "normal_zero_rate": float((normal["predicted_label"] == 0).mean()),
            "adversarial_zero_rate": float((adversarial["predicted_label"] == 0).mean()),
            "unknown_zero_rate": float(
                (prediction[prediction["inferred_kind"] == "unknown"]["predicted_label"] == 0).mean()
            ),
            "normal_spectral_view_agreement": float(
                (
                    normal["predicted_label"].to_numpy()
                    == spectral_prediction[
                        test_meta["inferred_kind"].eq("normal").to_numpy()
                    ]
                ).mean()
            ),
        }
    raw = candidates["v5a_raw"].set_index("sample_id")["predicted_label"]
    tlc = candidates["v5a_tlc"].set_index("sample_id")["predicted_label"]
    normal_boolean = test_meta["inferred_kind"].eq("normal").to_numpy()
    summary["raw_vs_tlc"] = {
        "all_agreement": float((raw == tlc).mean()),
        "normal_agreement": float(
            (
                raw[normal_boolean]
                == tlc[normal_boolean]
            ).mean()
        ),
    }
    v4 = pd.read_csv(REPORT_DIR / "final_predictions_v4.csv").set_index("sample_id")
    for name in candidates:
        v5 = candidates[name].set_index("sample_id")
        summary[f"{name}_vs_v4"] = {
            "all_agreement": float((v4["predicted_label"] == v5["predicted_label"]).mean()),
            "normal_agreement": float(
                (
                    v4.loc[normal_boolean, "predicted_label"]
                    == v5.loc[normal_boolean, "predicted_label"]
                ).mean()
            ),
        }
    with (REPORT_DIR / "v5a_summary.json").open("w", encoding="utf-8") as file_pointer:
        json.dump(summary, file_pointer, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
