#!/usr/bin/env python3
"""Inspect action-state feature slices for C3el exact-state labels.

This is a diagnostic, not a model trainer.  It joins the exact-state action
labels to the pre-action portfolio/candidate feature rows and searches for
simple feature slices that may explain why the strict C3el boundary contains
both large winners and large losers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEYS = ["timestamp", "strategy_id", "action_value"]
DEFAULT_LABELS = Path("data_perp/reports/c3el_exact_score_boundary_audit_20260628/combined_exact_labels_with_scores.csv")
DEFAULT_LAST4W_FEATURES = Path(
    "data_perp/reports/exact_state_size_action_learning_20260628_last4w_c3el_head_specific_scorer_threshold070/"
    "action_feature_rows.parquet"
)
DEFAULT_MAY_FEATURES = Path(
    "data_perp/reports/exact_state_size_action_learning_20260628_c3el_live_replay_training_panel_may06_may29/"
    "action_feature_rows.parquet"
)
DEFAULT_OUT_DIR = Path("data_perp/reports/c3el_exact_feature_slice_audit_20260628")


def _normalise_label_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.loc[out["timestamp"].notna()].copy()
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["action_value"] = pd.to_numeric(out.get("action_value", 0.0), errors="coerce").fillna(0.0).round(6)
    return out


def _normalise_feature_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.loc[out["timestamp"].notna()].copy()
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["action_value"] = pd.to_numeric(out.get("multiplier", 0.0), errors="coerce").fillna(0.0).round(6)
    return out


def load_joined(labels_path: Path, feature_paths: list[Path]) -> pd.DataFrame:
    labels = _normalise_label_keys(pd.read_csv(labels_path))
    features = [_normalise_feature_keys(pd.read_parquet(path)) for path in feature_paths]
    feature_frame = pd.concat(features, ignore_index=True, sort=False)
    feature_frame = feature_frame.drop_duplicates(KEYS, keep="last")
    feature_cols = [
        c
        for c in feature_frame.columns
        if c not in set(KEYS + ["multiplier"])
        and c not in labels.columns
        and pd.api.types.is_numeric_dtype(feature_frame[c])
    ]
    joined = labels.merge(feature_frame[KEYS + feature_cols], on=KEYS, how="left", validate="many_to_one")
    joined["feature_row_matched"] = joined[feature_cols].notna().any(axis=1) if feature_cols else False
    return joined


def _feature_candidates(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "delta_full_J",
        "delta_immediate_J",
        "delta_full_net_pnl",
        "delta_full_cost_pnl",
        "delta_full_turnover",
        "direct_delta_net_pnl",
        "baseline_net_pnl",
        "candidate_net_pnl",
        "base_full_J",
        "action_full_J",
        "base_immediate_J",
        "action_immediate_J",
        "base_full_net_pnl",
        "action_full_net_pnl",
        "base_full_cost_pnl",
        "action_full_cost_pnl",
        "base_full_turnover",
        "action_full_turnover",
        "base_immediate_trades",
        "action_immediate_trades",
    }
    out = []
    for col in frame.columns:
        if col in excluded or col in {"timestamp", "strategy_id", "source", "bucket", "day", "action_family"}:
            continue
        # This ratio explodes when there is no open notional and is not a
        # meaningful action-quality signal in the current panels.
        if col.endswith("_to_open_notional"):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]) and frame[col].nunique(dropna=True) >= 3:
            out.append(col)
    return out


def _summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "pos_share": np.nan,
            "sum_delta_full_J": 0.0,
            "mean_delta_full_J": np.nan,
            "median_delta_full_J": np.nan,
            "worst_delta_full_J": np.nan,
        }
    return {
        "rows": int(len(frame)),
        "pos_share": float(frame["delta_full_J"].gt(0.0).mean()),
        "sum_delta_full_J": float(frame["delta_full_J"].sum()),
        "mean_delta_full_J": float(frame["delta_full_J"].mean()),
        "median_delta_full_J": float(frame["delta_full_J"].median()),
        "worst_delta_full_J": float(frame["delta_full_J"].min()),
    }


def _candidate_slices(frame: pd.DataFrame, feature: str, min_rows: int) -> list[dict[str, Any]]:
    vals = pd.to_numeric(frame[feature], errors="coerce")
    valid = frame.loc[vals.notna()].copy()
    if len(valid) < max(2 * min_rows, 6) or vals.nunique(dropna=True) < 3:
        return []
    out: list[dict[str, Any]] = []
    for q_name, q in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
        threshold = float(valid[feature].quantile(q))
        for direction, mask in [
            ("low", valid[feature].le(threshold)),
            ("high", valid[feature].ge(threshold)),
        ]:
            kept = valid.loc[mask].copy()
            rejected = valid.loc[~mask].copy()
            if len(kept) < min_rows or len(rejected) < min_rows:
                continue
            kept_summary = _summary(kept)
            rejected_summary = _summary(rejected)
            base_summary = _summary(valid)
            out.append(
                {
                    "feature": feature,
                    "direction": direction,
                    "quantile": q_name,
                    "threshold": threshold,
                    "base_rows": base_summary["rows"],
                    "kept_rows": kept_summary["rows"],
                    "rejected_rows": rejected_summary["rows"],
                    "kept_pos_share": kept_summary["pos_share"],
                    "rejected_pos_share": rejected_summary["pos_share"],
                    "kept_sum_delta_full_J": kept_summary["sum_delta_full_J"],
                    "rejected_sum_delta_full_J": rejected_summary["sum_delta_full_J"],
                    "kept_mean_delta_full_J": kept_summary["mean_delta_full_J"],
                    "rejected_mean_delta_full_J": rejected_summary["mean_delta_full_J"],
                    "kept_median_delta_full_J": kept_summary["median_delta_full_J"],
                    "rejected_median_delta_full_J": rejected_summary["median_delta_full_J"],
                    "kept_worst_delta_full_J": kept_summary["worst_delta_full_J"],
                    "rejected_worst_delta_full_J": rejected_summary["worst_delta_full_J"],
                    "mean_lift": kept_summary["mean_delta_full_J"] - rejected_summary["mean_delta_full_J"],
                    "median_lift": kept_summary["median_delta_full_J"] - rejected_summary["median_delta_full_J"],
                }
            )
    return out


def _feature_stats(frame: pd.DataFrame, feature: str) -> dict[str, Any] | None:
    work = frame[[feature, "delta_full_J"]].dropna()
    if len(work) < 6 or work[feature].nunique() < 3:
        return None
    positive = work.loc[work["delta_full_J"].gt(0.0), feature]
    negative = work.loc[~work["delta_full_J"].gt(0.0), feature]
    if positive.empty or negative.empty:
        pos_minus_neg = np.nan
    else:
        pos_minus_neg = float(positive.median() - negative.median())
    return {
        "feature": feature,
        "rows": int(len(work)),
        "spearman_delta": float(work[feature].corr(work["delta_full_J"], method="spearman")),
        "median_positive": float(positive.median()) if not positive.empty else np.nan,
        "median_non_positive": float(negative.median()) if not negative.empty else np.nan,
        "median_pos_minus_nonpos": pos_minus_neg,
        "q25": float(work[feature].quantile(0.25)),
        "q50": float(work[feature].quantile(0.50)),
        "q75": float(work[feature].quantile(0.75)),
    }


def analyze_slice(frame: pd.DataFrame, *, name: str, min_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = _feature_candidates(frame)
    stats = [_feature_stats(frame, feature) for feature in features]
    stats_df = pd.DataFrame([x for x in stats if x is not None])
    slices: list[dict[str, Any]] = []
    for feature in features:
        slices.extend(_candidate_slices(frame, feature, min_rows=min_rows))
    slices_df = pd.DataFrame(slices)
    if not stats_df.empty:
        stats_df.insert(0, "sample", name)
        stats_df["abs_spearman_delta"] = stats_df["spearman_delta"].abs()
        stats_df = stats_df.sort_values("abs_spearman_delta", ascending=False)
    if not slices_df.empty:
        slices_df.insert(0, "sample", name)
        slices_df["defensive_success"] = -slices_df["rejected_sum_delta_full_J"] + slices_df["kept_sum_delta_full_J"].clip(upper=0.0)
        slices_df = slices_df.sort_values(
            ["kept_sum_delta_full_J", "kept_mean_delta_full_J", "kept_pos_share"],
            ascending=[False, False, False],
        )
    return stats_df, slices_df


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def write_report(joined: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out_dir / "exact_labels_with_action_features.csv", index=False)

    strict = joined.loc[joined["bucket"].eq("p80_d320")].copy()
    all_stats, all_slices = analyze_slice(joined, name="all_exact_labels", min_rows=8)
    strict_stats, strict_slices = analyze_slice(strict, name="strict_p80_d320", min_rows=5)

    stats_df = pd.concat([all_stats, strict_stats], ignore_index=True, sort=False)
    slices_df = pd.concat([all_slices, strict_slices], ignore_index=True, sort=False)
    stats_df.to_csv(out_dir / "feature_correlations.csv", index=False)
    slices_df.to_csv(out_dir / "feature_slice_candidates.csv", index=False)

    manifest = {
        "generated_by": "analyze_c3el_exact_feature_slices",
        "rows": int(len(joined)),
        "feature_rows_matched": int(joined["feature_row_matched"].sum()),
        "strict_p80_d320_rows": int(len(strict)),
        "outputs": {
            "joined": str(out_dir / "exact_labels_with_action_features.csv"),
            "correlations": str(out_dir / "feature_correlations.csv"),
            "slices": str(out_dir / "feature_slice_candidates.csv"),
            "summary": str(out_dir / "summary.md"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))

    lines = [
        "# C3el Exact-State Feature Slice Audit",
        "",
        "This report joins exact-state labels to action-state features and searches for simple explanatory slices.",
        "",
        f"Rows: `{len(joined)}`",
        f"Feature rows matched: `{int(joined['feature_row_matched'].sum())}`",
        f"Strict p80/d320 rows: `{len(strict)}`",
        "",
        "## Strongest Feature Correlations: All Labels",
        "",
    ]
    if all_stats.empty:
        lines.append("No usable numeric feature correlations.")
    else:
        cols = [
            "sample",
            "feature",
            "rows",
            "spearman_delta",
            "median_positive",
            "median_non_positive",
            "median_pos_minus_nonpos",
        ]
        lines.append(all_stats[cols].head(20).to_markdown(index=False, floatfmt=".4f"))
    lines.extend(["", "## Strongest Feature Correlations: Strict p80/d320", ""])
    if strict_stats.empty:
        lines.append("No usable strict p80/d320 numeric feature correlations.")
    else:
        cols = [
            "sample",
            "feature",
            "rows",
            "spearman_delta",
            "median_positive",
            "median_non_positive",
            "median_pos_minus_nonpos",
        ]
        lines.append(strict_stats[cols].head(20).to_markdown(index=False, floatfmt=".4f"))
    lines.extend(["", "## Best Simple Slices: All Labels", ""])
    if all_slices.empty:
        lines.append("No usable feature slices.")
    else:
        cols = [
            "sample",
            "feature",
            "direction",
            "quantile",
            "threshold",
            "kept_rows",
            "rejected_rows",
            "kept_pos_share",
            "kept_sum_delta_full_J",
            "rejected_sum_delta_full_J",
            "mean_lift",
            "kept_worst_delta_full_J",
        ]
        lines.append(all_slices[cols].head(20).to_markdown(index=False, floatfmt=".4f"))
    lines.extend(["", "## Best Simple Slices: Strict p80/d320", ""])
    if strict_slices.empty:
        lines.append("No usable strict p80/d320 feature slices.")
    else:
        cols = [
            "sample",
            "feature",
            "direction",
            "quantile",
            "threshold",
            "kept_rows",
            "rejected_rows",
            "kept_pos_share",
            "kept_sum_delta_full_J",
            "rejected_sum_delta_full_J",
            "mean_lift",
            "kept_worst_delta_full_J",
        ]
        lines.append(strict_slices[cols].head(20).to_markdown(index=False, floatfmt=".4f"))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- These slices are explanatory diagnostics only. They are small-sample and should not be promoted as hard gates.",
            "- Useful candidates should be treated as hypotheses for future exact-state label collection or predeclared shadow monitoring.",
            "- A feature slice is more credible if it improves the strict p80/d320 subset, not just the mixed all-label sample.",
            "",
            "## Current Hypotheses",
            "",
            "The strict p80/d320 state appears most useful when the size cut removes a concentrated short_asset opportunity set without broad portfolio/cooldown congestion.",
            "",
            "Predeclared monitoring candidates:",
            "",
            "1. `p80_d320 AND cooldown_count <= 38.5`",
            "2. `p80_d320 AND timestamp_rank_q90 <= 0.8641`",
            "3. `p80_d320 AND strategy_candidate_open_or_cooldown_symbol_share <= 0.3949`",
            "4. `p80_d320 AND strategy_rank_max <= 0.9054`",
            "",
            "These should be monitored as exact-state label collection rules, not deployed as live hard gates. The current sample is only 28 strict rows.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--last4w-features", type=Path, default=DEFAULT_LAST4W_FEATURES)
    parser.add_argument("--may-features", type=Path, default=DEFAULT_MAY_FEATURES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    joined = load_joined(args.labels, [args.last4w_features, args.may_features])
    manifest = write_report(joined, args.out_dir)
    print((args.out_dir / "summary.md").read_text())
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
