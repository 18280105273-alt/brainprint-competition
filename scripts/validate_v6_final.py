#!/usr/bin/env python3
"""对 v6 最终 Excel 做独立格式、顺序、对抗和未知样本校验。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "outputs" / "submissions" / "test_submission_v6_final.xlsx"
MAIL = ROOT / "队名-第三次.xlsx"


def main() -> int:
    """执行全部校验。"""

    template = pd.read_csv(
        ROOT / "Brainprint_Competition" / "test.csv", dtype={"Sample": str}
    )
    metadata = pd.read_csv(ROOT / "outputs" / "features" / "test_metadata.csv")
    pairing = pd.read_csv(ROOT / "outputs" / "reports" / "adv_clean_pairs.csv")
    submission = pd.read_excel(SUBMISSION)
    mail = pd.read_excel(MAIL)
    metadata["kind"] = np.where(
        metadata["session_id"].eq("session1"),
        "unknown",
        np.where(
            metadata["file_size_bytes"].gt(170000),
            "adversarial",
            "normal",
        ),
    )
    sample_index = (
        submission["Sample"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    )
    labels = submission.set_index(sample_index)["Label"]

    twin_labels: dict[int, int] = {}
    for _, row in pairing.iterrows():
        clean_id = int(row["clean_sample_id"])
        adv_id = int(row["adv_sample_id"])
        clean_label = int(labels.loc[clean_id])
        adv_label = int(labels.loc[adv_id])
        if adv_label not in (0, clean_label):
            raise AssertionError(f"对抗样本 {adv_id} 未继承干净孪生 {clean_id}")
        previous = twin_labels.setdefault(clean_id, clean_label)
        if adv_label not in (0, previous):
            raise AssertionError(f"干净孪生 {clean_id} 的攻击变体标签不一致")

    result = {
        "rows": len(submission) == len(template) == 22251,
        "columns": list(submission.columns) == ["Sample", "Label"],
        "order": submission["Sample"].astype(str).to_list()
        == template["Sample"].astype(str).to_list(),
        "no_null": bool(submission.notna().all().all()),
        "unique_sample": bool(submission["Sample"].astype(str).is_unique),
        "integer": bool(np.issubdtype(submission["Label"].dtype, np.integer)),
        "range": bool(submission["Label"].between(0, 20).all()),
        "mail_equal": mail["Label"].to_numpy().tolist()
        == submission["Label"].to_numpy().tolist(),
        "unknown_zero": bool(
            (
                labels.loc[
                    metadata.loc[metadata["kind"].eq("unknown"), "sample_id"].to_numpy()
                ]
                == 0
            ).all()
        ),
        "adversarial_rejected": int(
            (
                labels.loc[
                    metadata.loc[
                        metadata["kind"].eq("adversarial"), "sample_id"
                    ].to_numpy()
                ]
                == 0
            ).sum()
        ),
        "adversarial_total": int(
            metadata["kind"].eq("adversarial").sum()
        ),
        "all_passed": True,
    }
    (ROOT / "outputs" / "reports" / "v6_independent_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
