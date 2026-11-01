"""项目任务与输出文件目录。"""

TASKS = ("EC", "EO", "Imagery", "SSVEP")

TASK_NAMES = {
    "EC": "闭眼静息态",
    "EO": "睁眼静息态",
    "Imagery": "数字图像想象",
    "SSVEP": "稳态视觉诱发电位",
}

ARTIFACTS = {
    "audit": "outputs/audit/audit_summary.json",
    "features": "outputs/features/train_features.npz",
    "covariances": "outputs/features/train_covariances.npy",
    "metrics": "outputs/reports/final_tangent_metrics.csv",
    "submission": "outputs/submissions/test_submission_v3.xlsx",
}
