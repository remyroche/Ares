#!/usr/bin/env python3
"""Diagnose feature differences around exact-state size-action oracle positives.

The exact-state replay has shown that useful size interventions are sparse.
This script compares the groups that the safety oracle would intervene on
against learned-selector false positives and missed positives, using only the
artifact files already produced by run_exact_state_size_action_learning.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


KEY_COLS = ["fold_id", "timestamp", "strategy_id"]
DEFAULT_RUN_DIR = Path(
    "data_perp/reports/"
    "exact_state_size_action_learning_20260626_8fold_train480_eval96_safety_oracle_group_selector_v3"
)
SAFETY_ORACLE_ARM = "C3bn_bagged_safety_stage1_oracle_zero_or_half_diagnostic"
LEARNED_ARMS = [
    "C3bo_bagged_safety_stage1_learned_zero_or_half_gate",
    "C3bp_bagged_safety_stage1_bagged_zero_or_half_gate",
    "C3bq_bagged_safety_stage1_calibrated_zero_or_half_gate",
    "C3br_bagged_safety_stage1_oracle_group_zero_or_half_diagnostic",
    "C3bs_bagged_safety_stage1_diagnostic_veto_zero_or_half_gate",
    "C3bt_bagged_safety_stage1_hard_portfolio_veto_zero_or_half_gate",
    "C3bu_bagged_safety_stage1_learned_no_cut_risk_veto_zero_or_half_gate",
    "C3bv_bagged_safety_stage1_hard_portfolio_precondition_zero_or_half_gate",
    "C3bw_bagged_safety_stage1_fold_fitted_precondition_zero_or_half_gate",
    "C3bx_bagged_safety_stage1_direct_fixed_zero_or_half_top_gate",
    "C3by_bagged_safety_stage1_oracle_group_predicted_action_gate",
    "C3bz_bagged_safety_stage1_oracle_group_conservative_predicted_action_gate",
    "C3ca_bagged_safety_stage1_oracle_group_ranker_action_gate",
    "C3cb_bagged_safety_stage1_ranker_action_gate",
    "C3cc_bagged_safety_stage1_ranker_action_acceptance_gate",
    "C3cd_bagged_safety_stage1_oracle_group_recall_ranker_action_gate",
    "C3ce_bagged_safety_stage1_oracle_group_recall_ranker_acceptance_gate",
    "C3cf_bagged_safety_stage1_oracle_group_recall_economic_action_gate",
    "C3cg_bagged_safety_stage1_oracle_group_recall_group_action_gate",
    "C3ch_bagged_safety_stage1_oracle_group_recall_group_action_acceptance_gate",
    "C3ci_bagged_safety_stage1_decomposed_value_acceptance_gate",
    "C3cj_bagged_safety_stage1_decomposed_calibrated_gate",
    "C3ck_bagged_safety_stage1_decomposed_calibrated_acceptance_gate",
    "C3cl_stage1_calibrated_fixed_half_gate",
    "C3cm_stage1_calibrated_fixed_zero_gate",
    "C3cn_bagged_safety_stage1_oracle_group_recall_group_action_family_acceptance_gate",
    "C3co_bagged_safety_stage1_oracle_group_recall_high_p_group_action_family_gate",
    "C3cp_bagged_safety_stage1_oracle_group_recall_high_p_group_action_ranker_gate",
    "C3cq_bagged_safety_stage1_direct_group_action_family_gate",
    "C3cr_bagged_safety_stage1_direct_group_action_family_acceptance_gate",
    "C3cs_bagged_safety_stage1_direct_group_action_family_zero_acceptance_gate",
    "C3ct_bagged_safety_stage1_direct_group_action_family_zero_strict_acceptance_gate",
    "C3cu_bagged_safety_stage1_direct_group_action_family_zero_consensus_acceptance_gate",
    "C3cv_bagged_safety_stage1_direct_group_action_family_zero_value_consensus_gate",
    "C3cw_bagged_safety_stage1_direct_group_action_family_value_consensus_gate",
    "C3cx_bagged_safety_stage1_high_conf_multiplier_value_gate",
    "C3cy_bagged_safety_stage1_family_supported_multiplier_value_gate",
    "C3cz_bagged_safety_stage1_family_supported_multiplier_value_v2_gate",
    "C3da_bagged_safety_stage1_c3cw_or_c3cz_union_gate",
    "C3db_bagged_safety_stage1_c3cw_or_c3cz_nonoverlap_union_gate",
]
LABEL_COLS = {
    "multiplier",
    "action_binds",
    "delta_full_J",
    "delta_immediate_J",
    "delta_full_net_pnl",
    "delta_full_cost_pnl",
    "delta_full_turnover",
    "delta_full_J_per_notional",
    "delta_immediate_J_per_notional",
    "split",
    "best_multiplier",
    "best_gain",
    "best_margin",
    "best_gain_per_notional",
    "best_margin_per_notional",
    "best_immediate_gain",
    "best_nonbaseline_gain",
    "worst_nonbaseline_gain",
    "best_nonbaseline_multiplier",
    "group_can_bind",
    "y_intervene",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "fold_id" in out.columns:
        out["fold_id"] = pd.to_numeric(out["fold_id"], errors="coerce").astype("Int64")
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").astype(str)
    if "strategy_id" in out.columns:
        out["strategy_id"] = out["strategy_id"].astype(str)
    return out


def _baseline_group_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = _normalize_keys(panel)
    base = panel[np.isclose(pd.to_numeric(panel["multiplier"], errors="coerce"), 1.0)].copy()
    if base.duplicated(KEY_COLS).any():
        base = base.sort_values(KEY_COLS).drop_duplicates(KEY_COLS, keep="last")
    return base


def _bagged_stage1_features(stage1: pd.DataFrame) -> pd.DataFrame:
    stage1 = _normalize_keys(stage1)
    if "score_source" in stage1.columns and (stage1["score_source"] == "bagged_stage1").any():
        stage1 = stage1[stage1["score_source"] == "bagged_stage1"].copy()
    keep = [
        c
        for c in stage1.columns
        if c in KEY_COLS
        or c
        in {
            "p_intervene",
            "p_intervene_raw",
            "bag_vote_share",
            "bag_p_mean",
            "bag_p_std",
            "bag_model_count",
            "group_can_bind",
            "best_gain",
            "best_margin",
            "best_multiplier",
            "best_nonbaseline_gain",
            "group_affected_notional",
        }
    ]
    stage1 = stage1[keep].copy()
    rename = {
        c: f"stage1_{c}"
        for c in stage1.columns
        if c not in KEY_COLS and not c.startswith("stage1_")
    }
    return stage1.rename(columns=rename)


def _arm_flags(transfer: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    transfer = _normalize_keys(transfer)
    transfer["selected"] = transfer["selected"].astype(bool)
    transfer["selected_delta_full_J"] = pd.to_numeric(
        transfer["selected_delta_full_J"], errors="coerce"
    ).fillna(0.0)
    for col in [
        "p_intervene",
        "selection_score",
        "selected_multiplier",
        "selected_delta_immediate_J",
        "oracle_best_multiplier",
        "oracle_best_delta_full_J",
        "oracle_best_delta_immediate_J",
        "missed_positive_oracle",
    ]:
        if col not in transfer.columns:
            transfer[col] = np.nan

    oracle = transfer[transfer["arm"] == SAFETY_ORACLE_ARM].copy()
    oracle["oracle_positive"] = oracle["selected"] & (oracle["selected_delta_full_J"] > 0)
    oracle_keep = oracle[
        KEY_COLS
        + [
            "selected",
            "selected_multiplier",
            "selected_delta_full_J",
            "selected_delta_immediate_J",
            "oracle_best_multiplier",
            "oracle_best_delta_full_J",
            "oracle_best_delta_immediate_J",
            "missed_positive_oracle",
            "selection_score",
            "p_intervene",
        ]
    ].rename(
        columns={
            "selected": "safety_oracle_selected",
            "selected_multiplier": "safety_oracle_multiplier",
            "selected_delta_full_J": "safety_oracle_delta_full_J",
            "selected_delta_immediate_J": "safety_oracle_delta_immediate_J",
            "oracle_best_multiplier": "full_oracle_best_multiplier",
            "oracle_best_delta_full_J": "full_oracle_best_delta_full_J",
            "oracle_best_delta_immediate_J": "full_oracle_best_delta_immediate_J",
            "selection_score": "safety_oracle_selection_score",
            "p_intervene": "safety_oracle_p_intervene",
        }
    )
    oracle_keep["oracle_positive"] = oracle["oracle_positive"].to_numpy()

    learned = transfer[transfer["arm"].isin(LEARNED_ARMS)].copy()
    learned["learned_selected_positive"] = learned["selected"] & (
        learned["selected_delta_full_J"] > 0
    )
    learned["learned_selected_nonpositive"] = learned["selected"] & (
        learned["selected_delta_full_J"] <= 0
    )
    learned_agg = learned.groupby(KEY_COLS, dropna=False).agg(
        learned_selected_any=("selected", "max"),
        learned_true_positive=("learned_selected_positive", "max"),
        learned_false_positive=("learned_selected_nonpositive", "max"),
        learned_selected_count=("selected", "sum"),
        learned_positive_count=("learned_selected_positive", "sum"),
        learned_false_count=("learned_selected_nonpositive", "sum"),
        learned_best_selected_delta_full_J=("selected_delta_full_J", "max"),
        learned_worst_selected_delta_full_J=("selected_delta_full_J", "min"),
    )
    learned_agg = learned_agg.reset_index()
    return oracle_keep, learned_agg


def _numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    reserved = set(KEY_COLS) | LABEL_COLS | {
        "safety_oracle_selected",
        "safety_oracle_multiplier",
        "safety_oracle_delta_full_J",
        "safety_oracle_delta_immediate_J",
        "full_oracle_best_multiplier",
        "full_oracle_best_delta_full_J",
        "full_oracle_best_delta_immediate_J",
        "missed_positive_oracle",
        "safety_oracle_selection_score",
        "safety_oracle_p_intervene",
        "oracle_positive",
        "learned_selected_any",
        "learned_true_positive",
        "learned_false_positive",
        "learned_selected_count",
        "learned_positive_count",
        "learned_false_count",
        "learned_best_selected_delta_full_J",
        "learned_worst_selected_delta_full_J",
    }
    cols: list[str] = []
    for col in df.columns:
        if col in reserved:
            continue
        if col.startswith("stage1_best_") or col in {
            "stage1_y_intervene",
            "stage1_best_gain",
            "stage1_best_margin",
            "stage1_best_multiplier",
            "stage1_best_nonbaseline_gain",
        }:
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().sum() >= 10 and vals.nunique(dropna=True) > 1:
                cols.append(col)
    return cols


def _safe_stats(values: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {k: np.nan for k in ["n", "mean", "std", "q25", "median", "q75"]}
    return {
        "n": float(x.size),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=0)),
        "q25": float(x.quantile(0.25)),
        "median": float(x.median()),
        "q75": float(x.quantile(0.75)),
    }


def _contrast_rows(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    mask_a: pd.Series,
    mask_b: pd.Series,
    contrast_name: str,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for col in feature_cols:
        a = _safe_stats(df.loc[mask_a, col])
        b = _safe_stats(df.loc[mask_b, col])
        pooled = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        iqr = float(pooled.quantile(0.75) - pooled.quantile(0.25))
        scale = iqr / 1.349 if np.isfinite(iqr) and iqr > 0 else float(pooled.std(ddof=0))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = np.nan
        median_diff = a["median"] - b["median"]
        mean_diff = a["mean"] - b["mean"]
        rows.append(
            {
                "contrast": contrast_name,
                "feature": col,
                "n_a": a["n"],
                "n_b": b["n"],
                "median_a": a["median"],
                "median_b": b["median"],
                "median_diff": median_diff,
                "mean_a": a["mean"],
                "mean_b": b["mean"],
                "mean_diff": mean_diff,
                "robust_z_median_diff": median_diff / scale if np.isfinite(scale) else np.nan,
                "q25_a": a["q25"],
                "q75_a": a["q75"],
                "q25_b": b["q25"],
                "q75_b": b["q75"],
            }
        )
    return rows


def _write_report(
    out_path: Path,
    run_dir: Path,
    labeled: pd.DataFrame,
    contrast: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    counts = {
        "groups": int(len(labeled)),
        "oracle_positive": int(labeled["oracle_positive"].sum()),
        "learned_selected_any": int(labeled["learned_selected_any"].sum()),
        "learned_true_positive": int(labeled["learned_true_positive"].sum()),
        "learned_false_positive": int(labeled["learned_false_positive"].sum()),
        "missed_oracle_positive": int(labeled["missed_oracle_positive"].sum()),
        "feature_count": int(len(feature_cols)),
    }
    lines = [
        "# Size-Action Safety Oracle Feature Diagnostics",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Group Counts",
        "",
    ]
    for key, value in counts.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Strongest Contrasts", ""])
    for name in contrast["contrast"].dropna().unique():
        part = contrast[contrast["contrast"] == name].copy()
        part["abs_robust_z"] = part["robust_z_median_diff"].abs()
        top = part.sort_values("abs_robust_z", ascending=False).head(12)
        lines.append(f"### {name}")
        lines.append("")
        if top.empty:
            lines.append("_No usable contrast._")
            lines.append("")
            continue
        lines.append(
            "| feature | median A | median B | median diff | robust z | n A | n B |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in top.itertuples(index=False):
            lines.append(
                "| "
                f"{row.feature} | {row.median_a:.6g} | {row.median_b:.6g} | "
                f"{row.median_diff:.6g} | {row.robust_z_median_diff:.3f} | "
                f"{row.n_a:.0f} | {row.n_b:.0f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation Notes",
            "",
            "- `oracle_positive` means the bagged safety zero/half oracle selected a positive action.",
            "- `learned_false_positive` means at least one learned selector intervened with non-positive realized delta.",
            "- `missed_oracle_positive` means the safety oracle found a positive intervention and no learned arm selected a positive action for that group.",
            "- Large absolute robust-z contrasts are candidates for the compact sparse intervention feature set; they are not by themselves proof of causal usefulness.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir or (run_dir / "safety_oracle_feature_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = _read_csv(run_dir / "size_action_exact_panel.csv")
    stage1 = _read_csv(run_dir / "size_action_stage1_group_scores.csv")
    transfer = _read_csv(run_dir / "size_action_selector_transfer_diagnostics.csv")

    base = _baseline_group_features(panel)
    stage1_features = _bagged_stage1_features(stage1)
    oracle_flags, learned_flags = _arm_flags(transfer)

    # Transfer diagnostics are evaluation-only. Restrict the diagnostic universe
    # to those groups so train-panel rows do not dilute the background class.
    eval_keys = oracle_flags[KEY_COLS].drop_duplicates()
    labeled = base.merge(eval_keys, on=KEY_COLS, how="inner", validate="one_to_one")
    labeled = labeled.merge(stage1_features, on=KEY_COLS, how="left", validate="one_to_one")
    labeled = labeled.merge(oracle_flags, on=KEY_COLS, how="left", validate="one_to_one")
    labeled = labeled.merge(learned_flags, on=KEY_COLS, how="left", validate="one_to_one")

    bool_cols = [
        "safety_oracle_selected",
        "oracle_positive",
        "learned_selected_any",
        "learned_true_positive",
        "learned_false_positive",
    ]
    for col in bool_cols:
        if col in labeled.columns:
            labeled[col] = labeled[col].where(labeled[col].notna(), False).astype(bool)
    for col in [
        "learned_selected_count",
        "learned_positive_count",
        "learned_false_count",
    ]:
        if col in labeled.columns:
            labeled[col] = pd.to_numeric(labeled[col], errors="coerce").fillna(0).astype(int)

    labeled["missed_oracle_positive"] = labeled["oracle_positive"] & ~labeled[
        "learned_true_positive"
    ]
    labeled["background_no_action"] = ~labeled["oracle_positive"] & ~labeled[
        "learned_selected_any"
    ]

    feature_cols = _numeric_feature_cols(labeled)
    contrasts = []
    definitions = {
        "oracle_positive_vs_learned_false_positive": (
            labeled["oracle_positive"],
            labeled["learned_false_positive"],
        ),
        "learned_true_positive_vs_learned_false_positive": (
            labeled["learned_true_positive"],
            labeled["learned_false_positive"],
        ),
        "missed_oracle_positive_vs_background": (
            labeled["missed_oracle_positive"],
            labeled["background_no_action"],
        ),
        "oracle_positive_vs_background": (
            labeled["oracle_positive"],
            labeled["background_no_action"],
        ),
    }
    for name, (mask_a, mask_b) in definitions.items():
        if int(mask_a.sum()) == 0 or int(mask_b.sum()) == 0:
            continue
        contrasts.extend(_contrast_rows(labeled, feature_cols, mask_a, mask_b, name))
    contrast_df = pd.DataFrame(contrasts)
    if not contrast_df.empty:
        contrast_df["abs_robust_z_median_diff"] = contrast_df[
            "robust_z_median_diff"
        ].abs()
        contrast_df = contrast_df.sort_values(
            ["contrast", "abs_robust_z_median_diff"], ascending=[True, False]
        )

    candidate_cols = KEY_COLS + [
        "oracle_positive",
        "safety_oracle_selected",
        "safety_oracle_multiplier",
        "safety_oracle_delta_full_J",
        "full_oracle_best_multiplier",
        "full_oracle_best_delta_full_J",
        "learned_selected_any",
        "learned_true_positive",
        "learned_false_positive",
        "missed_oracle_positive",
        "learned_selected_count",
        "learned_positive_count",
        "learned_false_count",
        "stage1_p_intervene",
        "stage1_bag_vote_share",
        "stage1_bag_p_mean",
        "stage1_bag_p_std",
        "best_gain",
        "best_margin",
        "best_multiplier",
        "best_nonbaseline_gain",
        "group_can_bind",
    ]
    candidate_cols = [c for c in candidate_cols if c in labeled.columns]
    candidates = labeled[
        labeled["oracle_positive"]
        | labeled["learned_selected_any"]
        | labeled["missed_oracle_positive"]
    ][candidate_cols].sort_values(KEY_COLS)

    labeled.to_csv(out_dir / "safety_oracle_group_labeled_features.csv", index=False)
    contrast_df.to_csv(out_dir / "safety_oracle_feature_contrast.csv", index=False)
    candidates.to_csv(out_dir / "safety_oracle_candidate_groups.csv", index=False)
    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "groups": int(len(labeled)),
        "oracle_positive": int(labeled["oracle_positive"].sum()),
        "learned_selected_any": int(labeled["learned_selected_any"].sum()),
        "learned_true_positive": int(labeled["learned_true_positive"].sum()),
        "learned_false_positive": int(labeled["learned_false_positive"].sum()),
        "missed_oracle_positive": int(labeled["missed_oracle_positive"].sum()),
        "feature_count": int(len(feature_cols)),
    }
    (out_dir / "safety_oracle_feature_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_report(
        out_dir / "safety_oracle_feature_diagnostic_report.md",
        run_dir,
        labeled,
        contrast_df,
        feature_cols,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
