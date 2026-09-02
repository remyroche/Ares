#!/usr/bin/env python3
"""Audit direct accepted-frontier suppression actionability.

This is a lightweight artifact audit.  It does not train models or replay the
portfolio.  It explains why a direct-suppression learner can show useful OOF
signal while still failing promotion: the model may rank harmful accepted
frontier rows, but the threshold action must be recurrent, positive, and
deployable under the existing T1 contract.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/market_state_direct_suppression_actionability_audit")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _num(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._\n"
    view = frame.copy()
    if columns is not None:
        view = view.loc[:, [column for column in columns if column in view.columns]]
    view = view.head(max_rows)
    if view.empty:
        return "_No requested columns found._\n"
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in view.columns) + " |")
    return "\n".join(lines) + "\n"


def _coerce_policy_grid(grid: pd.DataFrame) -> pd.DataFrame:
    out = grid.copy()
    numeric_cols = [
        "probability_cutoff",
        "utility_cutoff",
        "max_delta",
        "suppressed_rows",
        "suppressed_unique_decision_keys",
        "loss_avoided",
        "winner_pnl_sacrificed",
        "defensive_success",
        "positive_fold_share",
        "valid_fold_count",
        "suppressed_folds",
        "suppression_fold_share",
        "mean_pred_prob",
        "mean_pred_utility",
        "selection_score",
    ]
    for column in numeric_cols:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    if "passes_diagnostic_gate" not in out.columns:
        out["passes_diagnostic_gate"] = False
    out["passes_diagnostic_gate"] = out["passes_diagnostic_gate"].astype(bool)
    return out


def _row_blockers(row: pd.Series, *, min_rows: int, min_folds: int) -> list[str]:
    blockers: list[str] = []
    if float(row.get("suppressed_rows", 0.0)) < float(min_rows):
        blockers.append("suppressed_rows_below_min")
    if float(row.get("suppressed_folds", 0.0)) < float(min_folds):
        blockers.append("suppressed_folds_below_min")
    if float(row.get("defensive_success", 0.0)) <= 0.0:
        blockers.append("defensive_success_not_positive")
    if float(row.get("loss_avoided", 0.0)) <= float(row.get("winner_pnl_sacrificed", 0.0)):
        blockers.append("loss_avoided_not_greater_than_winner_pnl_sacrificed")
    if float(row.get("positive_fold_share", 0.0)) < 0.5:
        blockers.append("positive_fold_share_below_50pct")
    return blockers


def audit_actionability(training_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _read_json(training_dir / "direct_suppression_training_summary.json")
    grid = _coerce_policy_grid(_read_csv(training_dir / "direct_suppression_policy_grid.csv"))
    folds = _read_csv(training_dir / "direct_suppression_fold_report.csv")
    feature_importance = _read_csv(training_dir / "direct_suppression_feature_importance.csv")

    policy = summary.get("policy_grid") if isinstance(summary.get("policy_grid"), dict) else {}
    min_rows = int(policy.get("min_suppressed_rows") or 2)
    min_folds = int(policy.get("min_suppressed_folds") or 2)

    if not grid.empty:
        blocker_lists = [_row_blockers(row, min_rows=min_rows, min_folds=min_folds) for _, row in grid.iterrows()]
        grid["diagnostic_blockers"] = [";".join(items) for items in blocker_lists]
        counts: Counter[str] = Counter()
        for items in blocker_lists:
            counts.update(items)
        positive = grid.loc[grid["suppressed_rows"].gt(0)].copy()
        recurrent = grid.loc[
            grid["suppressed_rows"].ge(min_rows)
            & grid["suppressed_folds"].ge(min_folds)
        ].copy()
    else:
        counts = Counter()
        positive = pd.DataFrame()
        recurrent = pd.DataFrame()

    if not grid.empty:
        scope_diag = (
            grid.groupby("policy_scope", dropna=False)
            .agg(
                policy_rows=("policy_scope", "size"),
                positive_suppression_rows=("suppressed_rows", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
                max_suppressed_rows=("suppressed_rows", "max"),
                max_suppressed_folds=("suppressed_folds", "max"),
                max_defensive_success=("defensive_success", "max"),
                max_positive_fold_share=("positive_fold_share", "max"),
                passing_rows=("passes_diagnostic_gate", "sum"),
            )
            .reset_index()
            .sort_values(
                ["passing_rows", "max_suppressed_folds", "max_defensive_success"],
                ascending=[False, False, False],
            )
        )
        top_rows = grid.sort_values(
            ["passes_diagnostic_gate", "selection_score", "suppressed_folds", "defensive_success"],
            ascending=[False, False, False, False],
        ).head(30)
    else:
        scope_diag = pd.DataFrame()
        top_rows = pd.DataFrame()

    if not feature_importance.empty:
        imp = feature_importance.copy()
        for column in ["classifier_importance", "regressor_importance"]:
            if column not in imp.columns:
                imp[column] = 0.0
            imp[column] = pd.to_numeric(imp[column], errors="coerce").fillna(0.0)
        imp["total_importance"] = imp["classifier_importance"] + imp["regressor_importance"]
        top_features = imp.sort_values("total_importance", ascending=False).head(15)
    else:
        top_features = pd.DataFrame()

    oof = summary.get("oof") if isinstance(summary.get("oof"), dict) else {}
    selection = summary.get("selection") if isinstance(summary.get("selection"), dict) else {}
    best_attempt = selection.get("best_attempt") if isinstance(selection.get("best_attempt"), dict) else {}
    max_suppressed_rows = int(grid["suppressed_rows"].max()) if not grid.empty else 0
    max_suppressed_folds = int(grid["suppressed_folds"].max()) if not grid.empty else 0
    max_defensive_success = _num(grid["defensive_success"].max(), 0.0) if not grid.empty else 0.0
    max_positive_fold_share = _num(grid["positive_fold_share"].max(), 0.0) if not grid.empty else 0.0
    max_recurrent_defensive_success = (
        _num(recurrent["defensive_success"].max(), 0.0) if not recurrent.empty else 0.0
    )
    max_recurrent_positive_fold_share = (
        _num(recurrent["positive_fold_share"].max(), 0.0) if not recurrent.empty else 0.0
    )
    passing_rows = int(grid["passes_diagnostic_gate"].sum()) if not grid.empty else 0

    if passing_rows > 0:
        dominant_blocker = None
        interpretation = "At least one direct-suppression policy row passed diagnostic gates."
    elif max_suppressed_rows < min_rows or max_suppressed_folds < min_folds:
        dominant_blocker = "insufficient_recurrent_action_support"
        interpretation = (
            "The learner ranks harmful accepted-frontier rows, but not enough "
            "suppression actions recur across folds to form a deployable threshold policy."
        )
    elif max_recurrent_defensive_success <= 0.0:
        dominant_blocker = "nonpositive_defensive_success"
        interpretation = (
            "The grid can create recurrent suppressions, but avoided losses do not "
            "exceed winner PnL sacrificed."
        )
    elif max_recurrent_positive_fold_share < 0.5:
        dominant_blocker = "nonrecurrent_positive_action_folds"
        interpretation = (
            "Some rows are suppressible across the minimum number of folds, but the "
            "positive action share is below 50%, so the apparent edge is not recurrent."
        )
    else:
        dominant_blocker = "diagnostic_gate_failed"
        interpretation = "No policy row passed all diagnostic gates."

    payload = {
        "generated_by": "audit_market_state_direct_suppression_actionability",
        "training_dir": str(training_dir),
        "output_dir": str(output_dir),
        "ledger_rows": summary.get("ledger_rows"),
        "unique_decision_keys": summary.get("unique_decision_keys"),
        "timestamp_count": summary.get("timestamp_count"),
        "active_heads": summary.get("active_heads"),
        "feature_count": summary.get("feature_count"),
        "oof_rows": oof.get("oof_rows"),
        "oof_unique_decision_keys": oof.get("oof_unique_decision_keys"),
        "oof_probability_auc": oof.get("prob_auc"),
        "oof_average_precision": oof.get("prob_average_precision"),
        "oof_utility_spearman": oof.get("utility_spearman"),
        "policy_grid_rows": int(len(grid)),
        "min_suppressed_rows": min_rows,
        "min_suppressed_folds": min_folds,
        "passing_policy_rows": passing_rows,
        "positive_suppression_policy_rows": int(len(positive)),
        "recurrent_support_policy_rows": int(len(recurrent)),
        "max_suppressed_rows": max_suppressed_rows,
        "max_suppressed_folds": max_suppressed_folds,
        "max_defensive_success": max_defensive_success,
        "max_positive_fold_share": max_positive_fold_share,
        "max_recurrent_defensive_success": max_recurrent_defensive_success,
        "max_recurrent_positive_fold_share": max_recurrent_positive_fold_share,
        "selected_arm": selection.get("selected_arm"),
        "selection_reason": selection.get("reason"),
        "best_attempt": best_attempt,
        "dominant_blocker": dominant_blocker,
        "blocker_counts": dict(sorted(counts.items())),
        "fold_valid_unique_key_min": int(pd.to_numeric(folds.get("valid_unique_decision_keys", pd.Series(dtype=float)), errors="coerce").min()) if not folds.empty else None,
        "fold_valid_unique_key_median": _num(pd.to_numeric(folds.get("valid_unique_decision_keys", pd.Series(dtype=float)), errors="coerce").median()) if not folds.empty else None,
        "interpretation": interpretation,
    }

    scope_path = output_dir / "direct_suppression_actionability_by_scope.csv"
    top_path = output_dir / "direct_suppression_top_policy_rows.csv"
    features_path = output_dir / "direct_suppression_top_features.csv"
    json_path = output_dir / "direct_suppression_actionability_audit.json"
    report_path = output_dir / "direct_suppression_actionability_audit.md"

    scope_diag.to_csv(scope_path, index=False)
    top_rows.to_csv(top_path, index=False)
    top_features.to_csv(features_path, index=False)
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_lines = [
        "# Direct Suppression Actionability Audit",
        "",
        "This audit reads existing direct-suppression training artifacts only. It does not train, score, or replay.",
        "",
        "## Verdict",
        "",
        f"- Selected arm: `{payload['selected_arm']}`",
        f"- Selection reason: `{payload['selection_reason']}`",
        f"- Dominant blocker: `{payload['dominant_blocker']}`",
        f"- Interpretation: {payload['interpretation']}",
        "",
        "## Signal Versus Actionability",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {
                        "oof_auc": payload["oof_probability_auc"],
                        "oof_ap": payload["oof_average_precision"],
                        "oof_utility_spearman": payload["oof_utility_spearman"],
                        "policy_rows": payload["policy_grid_rows"],
                        "passing_rows": payload["passing_policy_rows"],
                        "max_suppressed_rows": payload["max_suppressed_rows"],
                        "max_suppressed_folds": payload["max_suppressed_folds"],
                        "max_defensive_success": payload["max_defensive_success"],
                        "max_positive_fold_share": payload["max_positive_fold_share"],
                        "max_recurrent_defensive_success": payload["max_recurrent_defensive_success"],
                        "max_recurrent_positive_fold_share": payload["max_recurrent_positive_fold_share"],
                    }
                ]
            )
        ),
        "## Policy Scope Diagnostics",
        "",
        _markdown_table(scope_diag, max_rows=20),
        "## Top Policy Rows",
        "",
        _markdown_table(
            top_rows,
            columns=[
                "policy_scope",
                "controller_arm",
                "target_head",
                "probability_cutoff",
                "utility_cutoff",
                "max_delta",
                "suppressed_rows",
                "suppressed_folds",
                "loss_avoided",
                "winner_pnl_sacrificed",
                "defensive_success",
                "positive_fold_share",
                "diagnostic_blockers",
            ],
            max_rows=15,
        ),
        "## Top Features",
        "",
        _markdown_table(top_features, max_rows=15),
        "## Generated Files",
        "",
        f"- `{json_path}`",
        f"- `{report_path}`",
        f"- `{scope_path}`",
        f"- `{top_path}`",
        f"- `{features_path}`",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = audit_actionability(args.training_dir, args.output_dir)
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
