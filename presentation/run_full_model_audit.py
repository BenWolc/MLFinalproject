"""
Full Model Audit Runner

Put this file in the same presentation folder as:
  - compare_zip_project_models.py
  - compare_all_models.py
  - tech_employment_2000_2025.csv
  - layoffs.csv
  - your repo model files / preprocessed CSVs

Then run:
  python run_full_model_audit.py

What it does:
1. Runs compare_zip_project_models.py to evaluate your team's actual repo model files.
2. Runs compare_all_models.py to evaluate the broader model zoo / optimal models we tested.
3. Combines the output CSVs into one audit summary.
4. Writes a readable text report and combined CSVs.

Outputs:
  full_model_audit_summary.txt
  full_model_audit_leaderboard.csv
  full_model_audit_repo_regression.csv
  full_model_audit_repo_classification.csv
  full_model_audit_broad_walkforward.csv
  full_model_audit_broad_final_test.csv
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


ZIP_SCRIPT = "compare_zip_project_models.py"
BROAD_SCRIPT = "compare_all_models.py"


def run_script(script_name: str) -> bool:
    if not Path(script_name).exists():
        print(f"Skipping {script_name}: file not found.")
        return False

    print("\n" + "=" * 120)
    print(f"RUNNING {script_name}")
    print("=" * 120)

    result = subprocess.run(
        [sys.executable, script_name],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout)

    if result.returncode != 0:
        print(f"WARNING: {script_name} exited with code {result.returncode}")
        return False
    return True


def read_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    if Path(path).exists():
        try:
            return pd.read_csv(path)
        except Exception as exc:
            print(f"Could not read {path}: {exc}")
    return None


def fmt(x, ndigits: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return str(x)


def first_row(df: Optional[pd.DataFrame], sort_col: str, ascending: bool = True) -> Optional[pd.Series]:
    if df is None or df.empty or sort_col not in df.columns:
        return None
    good = df.dropna(subset=[sort_col]).copy()
    if good.empty:
        return None
    return good.sort_values(sort_col, ascending=ascending).iloc[0]


def copy_output(src: str, dst: str) -> None:
    df = read_csv_if_exists(src)
    if df is not None:
        df.to_csv(dst, index=False)


def build_leaderboard(
    repo_reg: Optional[pd.DataFrame],
    repo_cls: Optional[pd.DataFrame],
    broad_cv: Optional[pd.DataFrame],
    broad_final: Optional[pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    if repo_reg is not None and not repo_reg.empty:
        for _, r in repo_reg.iterrows():
            rows.append({
                "group": "repo_regression_models_actual_code",
                "evaluation": "repo_test_split",
                "model": r.get("model"),
                "source": r.get("source_file"),
                "primary_metric": "MAE_pct_points",
                "primary_metric_value": r.get("MAE_pct_points"),
                "RMSE_pct_points": r.get("RMSE_pct_points"),
                "R2": r.get("R2_pct_space"),
                "notes": "Actual project repo model logic evaluated on repo X/y split; target converted from log1p to percentage points.",
            })

    if repo_cls is not None and not repo_cls.empty:
        for _, r in repo_cls.iterrows():
            rows.append({
                "group": "repo_classification_models_actual_code",
                "evaluation": "repo_test_split_elevated_layoff",
                "model": r.get("model"),
                "source": r.get("source_file"),
                "primary_metric": "Test_ROC_AUC",
                "primary_metric_value": r.get("Test_ROC_AUC"),
                "RMSE_pct_points": np.nan,
                "R2": np.nan,
                "notes": f"Classification model at threshold {r.get('threshold_pct', 'N/A')}% layoffs; also inspect F1 due class imbalance.",
            })

    if broad_cv is not None and not broad_cv.empty:
        for _, r in broad_cv.iterrows():
            rows.append({
                "group": "broad_model_zoo_and_optimal_models",
                "evaluation": "walk_forward_cv_main_selection",
                "model": r.get("model"),
                "source": "compare_all_models.py",
                "primary_metric": "CV_MAE",
                "primary_metric_value": r.get("CV_MAE"),
                "RMSE_pct_points": r.get("CV_RMSE"),
                "R2": np.nan,
                "notes": "Broader model zoo / optimal models evaluated by historical walk-forward CV.",
            })

    if broad_final is not None and not broad_final.empty:
        for _, r in broad_final.iterrows():
            rows.append({
                "group": "broad_model_zoo_and_optimal_models",
                "evaluation": "final_test_estimated_targets_only",
                "model": r.get("model"),
                "source": "compare_all_models.py",
                "primary_metric": "MAE_pct_points",
                "primary_metric_value": r.get("MAE_pct_points"),
                "RMSE_pct_points": r.get("RMSE_pct_points"),
                "R2": r.get("R2_pct"),
                "notes": "Final test set has estimated targets only; use cautiously.",
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["group", "evaluation", "primary_metric_value"], ascending=[True, True, True])
    return out


def make_report(
    repo_reg: Optional[pd.DataFrame],
    repo_cls: Optional[pd.DataFrame],
    broad_cv: Optional[pd.DataFrame],
    broad_final: Optional[pd.DataFrame],
) -> str:
    lines = []
    lines.append("FULL MODEL AUDIT SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    best_repo_reg = first_row(repo_reg, "MAE_pct_points", True)
    best_repo_cls_auc = first_row(repo_cls, "Test_ROC_AUC", False)
    best_repo_cls_f1 = first_row(repo_cls, "Test_F1", False)
    best_broad_cv = first_row(broad_cv, "CV_MAE", True)
    best_broad_final = first_row(broad_final, "MAE_pct_points", True)

    lines.append("1) Best actual repo regression model")
    if best_repo_reg is not None:
        lines.append(f"   Model: {best_repo_reg.get('model')}")
        lines.append(f"   Source file: {best_repo_reg.get('source_file')}")
        lines.append(f"   MAE: {fmt(best_repo_reg.get('MAE_pct_points'))} percentage points")
        lines.append(f"   RMSE: {fmt(best_repo_reg.get('RMSE_pct_points'))} percentage points")
        lines.append(f"   Note: this is the actual project repo model logic on the repo's X/y split.")
    else:
        lines.append("   Not available.")
    lines.append("")

    lines.append("2) Best actual repo classification model")
    if best_repo_cls_auc is not None:
        lines.append(f"   Best ROC AUC model: {best_repo_cls_auc.get('model')}")
        lines.append(f"   ROC AUC: {fmt(best_repo_cls_auc.get('Test_ROC_AUC'))}")
        lines.append(f"   F1 for that model: {fmt(best_repo_cls_auc.get('Test_F1'))}")
    else:
        lines.append("   Not available.")
    if best_repo_cls_f1 is not None:
        lines.append(f"   Best F1 model: {best_repo_cls_f1.get('model')}")
        lines.append(f"   F1: {fmt(best_repo_cls_f1.get('Test_F1'))}")
    lines.append("   Note: high AUC but low F1 can happen when positives are rare; inspect precision/recall.")
    lines.append("")

    lines.append("3) Best broad / optimal model from the expanded model zoo")
    if best_broad_cv is not None:
        lines.append(f"   Model: {best_broad_cv.get('model')}")
        lines.append(f"   Walk-forward CV MAE: {fmt(best_broad_cv.get('CV_MAE'))} percentage points")
        lines.append(f"   Walk-forward CV RMSE: {fmt(best_broad_cv.get('CV_RMSE'))} percentage points")
        lines.append("   Note: this is the fairest result for time-based forecasting because it averages over many historical years.")
    else:
        lines.append("   Not available.")
    lines.append("")

    lines.append("4) Best model on final held-out test from broad zoo")
    if best_broad_final is not None:
        lines.append(f"   Model: {best_broad_final.get('model')}")
        lines.append(f"   MAE: {fmt(best_broad_final.get('MAE_pct_points'))} percentage points")
        lines.append("   Caveat: final test has only estimated target rows, so this may reward predicting the dataset's default estimated value.")
    else:
        lines.append("   Not available.")
    lines.append("")

    lines.append("5) Recommended interpretation")
    lines.append("   - Use the repo regression comparison to report how your team's actual implemented models perform.")
    lines.append("   - Use the broad walk-forward CV comparison to discuss whether simple historical models or complex ML generalize better.")
    lines.append("   - Do not rely only on the final 2023-2024 test set if it contains estimated labels only.")
    lines.append("   - If the repo XGBoost result is much better than walk-forward CV models, mention that the repo split/preprocessing differs and should be checked for leakage.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    print("Full model audit started.")
    print("Current folder:", os.getcwd())

    zip_ok = run_script(ZIP_SCRIPT)
    broad_ok = run_script(BROAD_SCRIPT)

    repo_reg = read_csv_if_exists("zip_regression_model_comparison.csv")
    repo_cls = read_csv_if_exists("zip_classification_model_comparison.csv")
    broad_cv = read_csv_if_exists("all_models_walk_forward_summary.csv")
    broad_final = read_csv_if_exists("all_models_final_test_results.csv")

    copy_output("zip_regression_model_comparison.csv", "full_model_audit_repo_regression.csv")
    copy_output("zip_classification_model_comparison.csv", "full_model_audit_repo_classification.csv")
    copy_output("all_models_walk_forward_summary.csv", "full_model_audit_broad_walkforward.csv")
    copy_output("all_models_final_test_results.csv", "full_model_audit_broad_final_test.csv")

    leaderboard = build_leaderboard(repo_reg, repo_cls, broad_cv, broad_final)
    leaderboard.to_csv("full_model_audit_leaderboard.csv", index=False)

    report = make_report(repo_reg, repo_cls, broad_cv, broad_final)
    Path("full_model_audit_summary.txt").write_text(report, encoding="utf-8")

    print("\n" + report)
    print("\nSaved:")
    print(" - full_model_audit_summary.txt")
    print(" - full_model_audit_leaderboard.csv")
    print(" - full_model_audit_repo_regression.csv")
    print(" - full_model_audit_repo_classification.csv")
    print(" - full_model_audit_broad_walkforward.csv")
    print(" - full_model_audit_broad_final_test.csv")

    if not zip_ok:
        print("\nWarning: repo model script did not fully run. Check compare_zip_project_models.py output above.")
    if not broad_ok:
        print("\nWarning: broad model script did not fully run. Check compare_all_models.py output above.")


if __name__ == "__main__":
    main()
