#!/usr/bin/env python3
"""Audit recurrent and conditionally recurrent F72 SHAP-derived outputs.

This is deliberately a diagnostic-only stage.  It consumes the immutable,
target-free F72 SHAP ledger first and opens rich-policy labels only after every
target-free receipt is verified.  It never fits a model, changes an admission
rule, or calculates CMI/IC for raw causal features: the only examined fields
are newly created ``shap_f72_*`` outputs.

The audit answers a narrower question than the original all-fold promotion
gate: whether a SHAP-derived output has a stable relationship with realised
policy net *within a comparable Base-score region*.  This catches outputs
whose utility is conditional on Base confidence rather than broad ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamp_weighted_mean(frame: pd.DataFrame, value: str) -> float:
    if frame.empty:
        return float("nan")
    return float(frame.groupby("__decision_ts__", sort=False)[value].mean().mean())


def _band_effect(
    frame: pd.DataFrame,
    feature_pct: str,
) -> tuple[float, int, int, int]:
    """Return high-minus-low policy bps in a fixed Base-rank band.

    Both high and low are defined after restricting to one fixed Base-rank
    band, then ranking within timestamp.  This asks whether the SHAP output
    retains information *after* Base confidence is held comparable.  Ranking
    across the full timestamp would leave no low observations for an output
    that is itself strongly aligned with the Base score.
    """

    scoped = frame.loc[frame[feature_pct].notna()].copy()
    high = scoped.loc[scoped[feature_pct] >= 0.75]
    low = scoped.loc[scoped[feature_pct] <= 0.25]
    high_by_ts = high.groupby("__decision_ts__", sort=False)["policy_net_bps"].mean()
    low_by_ts = low.groupby("__decision_ts__", sort=False)["policy_net_bps"].mean()
    paired = pd.concat([high_by_ts.rename("high"), low_by_ts.rename("low")], axis=1).dropna()
    if paired.empty:
        return float("nan"), 0, int(high.shape[0]), int(low.shape[0])
    return float((paired["high"] - paired["low"]).mean()), int(paired.shape[0]), int(high.shape[0]), int(low.shape[0])


def _directional_stats(values: pd.Series) -> tuple[str, int, float, float, float]:
    clean = values.dropna().astype(float)
    if clean.empty:
        return "unresolved", 0, float("nan"), float("nan"), float("nan")
    direction = "high" if clean.median() >= 0.0 else "low"
    aligned = clean if direction == "high" else -clean
    return (
        direction,
        int((aligned > 0.0).sum()),
        float(aligned.median()),
        float(aligned.mean()),
        float(aligned.quantile(0.25)),
    )


def _write_markdown(summary: pd.DataFrame, out: Path, shap_root: Path) -> None:
    strict = summary.loc[summary["tier"] == "strict_core"]
    recurrent = summary.loc[summary["tier"] == "recurrent"]
    conditional = summary.loc[summary["tier"] == "conditional"]
    rejected = summary.loc[summary["tier"] == "not_selected"]
    lines = [
        "# F72 SHAP-Derived Conditional Candidate Audit",
        "",
        "## Scope",
        "",
        "This is a diagnostic-only audit of newly derived `shap_f72_*` outputs from the immutable strict-OOF F72 ledger. It neither fits a new model nor changes the canonical P8U Router50 → F72 → Under F120 research contract.",
        "",
        f"Source ledger: `{shap_root}`.",
        "",
        "Raw causal fields never enter CMI/IC here. The prior CMI/IC evidence belongs only to SHAP-derived outputs. This audit adds timestamp-local high-versus-low realised rich-policy-net contrasts after all target-free score receipts were confirmed.",
        "",
        "## Interpretation",
        "",
        "`top30_effect` compares the top and bottom SHAP quartiles *within* the Base top-30% band at each timestamp, then averages timestamps equally. Thus it tests whether an output retains information after Base confidence is held comparable. A `high` orientation means a high SHAP value is better; `low` means the inverse. The sign is descriptive, selected across the strict OOF months only, and must be frozen before any later downstream test.",
        "",
        "Tiers are intentionally conservative:",
        "",
        "- `strict_core`: the original all-fold strong evidence standard.",
        "- `recurrent`: broad directional recurrence (at least 10/12 positive rank-IC folds) plus a recurrent Top-30 SHAP effect.",
        "- `conditional`: weaker broad ranking but a recurrent, economically material Top-30 conditional effect (at least 9/12 aligned folds and median aligned effect at least 15 bps).",
        "- `not_selected`: no reliable candidate under these relaxed screens.",
        "",
    ]
    for title, frame in [
        ("Strict core", strict),
        ("Recurrent candidates", recurrent),
        ("Conditional candidates", conditional),
    ]:
        lines.extend([f"## {title}", ""])
        if frame.empty:
            lines.extend(["None.", ""])
            continue
        cols = [
            "feature",
            "top30_orientation",
            "top30_aligned_folds",
            "top30_effect_median_bps",
            "top30_effect_q25_bps",
            "positive_ic_folds",
            "ts_ic_mean",
        ]
        lines.extend([
            "| Feature | Orientation | Aligned folds | Median Top-30 effect | Q25 effect | Positive IC folds | Mean IC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in frame.sort_values(["top30_effect_median_bps", "ts_ic_mean"], ascending=False).itertuples(index=False):
            values = {name: getattr(row, name) for name in cols}
            lines.append(
                "| {feature} | {top30_orientation} | {top30_aligned_folds}/12 | {top30_effect_median_bps:+.2f} | {top30_effect_q25_bps:+.2f} | {positive_ic_folds}/12 | {ts_ic_mean:+.4f} |".format(**values)
            )
        lines.append("")
    lines.extend([
        "## Decision",
        "",
        f"The audit classified {strict.shape[0]} strict-core, {recurrent.shape[0]} recurrent, {conditional.shape[0]} conditional, and {rejected.shape[0]} non-selected outputs. These are research candidates only. Any future F72/Under test must compare a frozen shortlist against the exact F72/F120 control on a later strict OOF block before it can alter the canonical contract.",
        "",
    ])
    (out / "F72_SHAP_CONDITIONAL_CANDIDATE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shap-root", required=True, type=Path)
    parser.add_argument("--policy-path", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--top-base-floor", type=float, default=0.70)
    parser.add_argument("--min-conditional-folds", type=int, default=9)
    parser.add_argument("--min-conditional-median-bps", type=float, default=15.0)
    args = parser.parse_args()

    root = args.shap_root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipts = pd.read_parquet(root / "target_free_receipts.parquet")
    required = [
        "train_labels_resolved_before_reserve",
        "target_free_persisted_before_metrics",
        "router_top50_identity_exact",
    ]
    if receipts.empty or not receipts[required].all().all():
        raise RuntimeError("F72 SHAP target-free lineage receipt is incomplete")
    months = receipts["held_month"].astype(str).tolist()
    frames: list[pd.DataFrame] = []
    for month in months:
        path = root / "target_free_shap_features" / f"month={month}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        if frame["candidate_id"].duplicated().any():
            raise RuntimeError(f"Duplicate target-free candidate IDs in {month}")
        frame["held_month"] = month
        frames.append(frame)
    target_free = pd.concat(frames, ignore_index=True)
    feature_columns = [column for column in target_free.columns if column.startswith("shap_f72_")]
    if not feature_columns:
        raise RuntimeError("No derived SHAP fields found")

    # Only after every target-free monthly receipt and feature panel is loaded
    # do we open resolved outcome labels for this analysis.
    policy = pd.read_parquet(
        args.policy_path,
        columns=["candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"],
    )
    policy = policy.loc[policy["policy_path_valid"].fillna(False)].copy()
    joined = target_free.merge(policy, on="candidate_id", how="inner", validate="one_to_one")
    if joined.empty:
        raise RuntimeError("No valid policy rows after target-free score join")
    if (pd.to_datetime(joined["policy_label_available_ts"], utc=True) <= pd.to_datetime(joined["__decision_ts__"], utc=True)).any():
        raise RuntimeError("Policy label is not resolved strictly after its decision timestamp")

    base = pd.to_numeric(joined["f72_base_rank_ts"], errors="coerce")
    bands = {
        "top30": base >= args.top_base_floor,
        "mid30": (base >= 0.40) & (base < args.top_base_floor),
        "lower40": base < 0.40,
    }
    evidence_rows: list[dict[str, object]] = []
    for held_month, month_frame in joined.groupby("held_month", sort=True):
        month_frame = month_frame.copy()
        for feature in feature_columns:
            for band, mask in bands.items():
                scoped = month_frame.loc[mask, ["__decision_ts__", "policy_net_bps", feature]].copy()
                pct_name = "__feature_pct__"
                values = pd.to_numeric(scoped[feature], errors="coerce")
                scoped[pct_name] = values.groupby(scoped["__decision_ts__"], sort=False).rank(method="average", pct=True)
                effect, paired_ts, high_rows, low_rows = _band_effect(scoped, pct_name)
                evidence_rows.append(
                    {
                        "held_month": held_month,
                        "feature": feature,
                        "base_band": band,
                        "high_minus_low_policy_net_bps": effect,
                        "paired_timestamps": paired_ts,
                        "high_rows": high_rows,
                        "low_rows": low_rows,
                    }
                )
    fold_effects = pd.DataFrame(evidence_rows)
    original = pd.read_parquet(root / "shap_derived_summary.parquet")
    rows: list[dict[str, object]] = []
    for feature, feature_effects in fold_effects.groupby("feature", sort=True):
        top30 = feature_effects.loc[feature_effects["base_band"] == "top30", "high_minus_low_policy_net_bps"]
        top30_direction, top30_folds, top30_median, top30_mean, top30_q25 = _directional_stats(top30)
        mid30_direction, mid30_folds, mid30_median, mid30_mean, mid30_q25 = _directional_stats(
            feature_effects.loc[feature_effects["base_band"] == "mid30", "high_minus_low_policy_net_bps"]
        )
        lower40_direction, lower40_folds, lower40_median, lower40_mean, lower40_q25 = _directional_stats(
            feature_effects.loc[feature_effects["base_band"] == "lower40", "high_minus_low_policy_net_bps"]
        )
        summary_row = original.loc[original["feature"] == feature]
        if summary_row.empty:
            raise RuntimeError(f"Missing original evidence summary for {feature}")
        s = summary_row.iloc[0]
        strict_core = bool(
            int(s["positive_ic_folds"]) == int(s["folds"])
            and float(s["ts_ic_min"]) >= 0.18
            and float(s["ts_top10_best_min"]) > 0.0
        )
        recurrent = bool(
            int(s["positive_ic_folds"]) >= 10
            and float(s["ts_ic_mean"]) >= 0.03
            and top30_folds >= args.min_conditional_folds
            and top30_median >= args.min_conditional_median_bps
        )
        conditional = bool(
            not recurrent
            and top30_folds >= args.min_conditional_folds
            and top30_median >= args.min_conditional_median_bps
            and top30_q25 > 0.0
        )
        tier = "strict_core" if strict_core else "recurrent" if recurrent else "conditional" if conditional else "not_selected"
        rows.append(
            {
                "feature": feature,
                "tier": tier,
                "top30_orientation": top30_direction,
                "top30_aligned_folds": top30_folds,
                "top30_effect_median_bps": top30_median,
                "top30_effect_mean_bps": top30_mean,
                "top30_effect_q25_bps": top30_q25,
                "mid30_orientation": mid30_direction,
                "mid30_aligned_folds": mid30_folds,
                "mid30_effect_median_bps": mid30_median,
                "mid30_effect_q25_bps": mid30_q25,
                "lower40_orientation": lower40_direction,
                "lower40_aligned_folds": lower40_folds,
                "lower40_effect_median_bps": lower40_median,
                "lower40_effect_q25_bps": lower40_q25,
                "cmi_median": float(s["cmi_median"]),
                "cmi_min": float(s["cmi_min"]),
                "ts_ic_mean": float(s["ts_ic_mean"]),
                "ts_ic_min": float(s["ts_ic_min"]),
                "positive_ic_folds": int(s["positive_ic_folds"]),
                "folds": int(s["folds"]),
                "stable_shap_signal": float(s["stable_shap_signal"]),
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        ["tier", "top30_effect_median_bps", "ts_ic_mean"],
        ascending=[True, False, False],
    )
    selected_features = summary.loc[summary["tier"] != "not_selected", "feature"].tolist()
    redundancy_rows: list[dict[str, object]] = []
    if len(selected_features) > 1:
        for held_month, month_frame in target_free.groupby("held_month", sort=True):
            corr = month_frame[selected_features].corr(method="spearman")
            for left_idx, left in enumerate(selected_features):
                for right in selected_features[left_idx + 1 :]:
                    redundancy_rows.append(
                        {
                            "held_month": held_month,
                            "feature_left": left,
                            "feature_right": right,
                            "spearman": float(corr.loc[left, right]),
                        }
                    )
    redundancy = pd.DataFrame(redundancy_rows)
    if not redundancy.empty:
        pair_summary = (
            redundancy.assign(abs_spearman=lambda frame: frame["spearman"].abs())
            .groupby(["feature_left", "feature_right"], as_index=False)
            .agg(
                median_spearman=("spearman", "median"),
                median_abs_spearman=("abs_spearman", "median"),
                max_abs_spearman=("abs_spearman", "max"),
            )
        )
    else:
        pair_summary = pd.DataFrame(columns=["feature_left", "feature_right", "median_spearman", "median_abs_spearman", "max_abs_spearman"])
    fold_effects.to_parquet(out / "shap_conditional_fold_effects.parquet", index=False)
    summary.to_parquet(out / "shap_conditional_candidates.parquet", index=False)
    redundancy.to_parquet(out / "shap_candidate_redundancy_by_fold.parquet", index=False)
    pair_summary.to_parquet(out / "shap_candidate_redundancy_summary.parquet", index=False)
    _write_markdown(summary, out, root)
    correctness = {
        "schema": "strict_r3_p8u_f72_shap_conditional_candidates_v1",
        "scope": "offline post-score SHAP-derived diagnostics only; no model/admission/portfolio/live mutation",
        "raw_feature_cmi_or_ic_not_run": True,
        "all_assessed_features_are_new_shap_derived": bool(all(f.startswith("shap_f72_") for f in feature_columns)),
        "all_target_free_receipts_verified_before_policy_open": True,
        "all_policy_labels_resolve_after_decision": True,
        "router_identity_receipts_exact": bool(receipts["router_top50_identity_exact"].all()),
        "folds": months,
        "n_features": len(feature_columns),
        "rows_with_valid_policy": int(joined.shape[0]),
    }
    (out / "correctness_report.json").write_text(json.dumps(correctness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "strict_r3_p8u_f72_shap_conditional_candidates_v1",
        "shap_root": str(root),
        "shap_root_receipts_sha256": _sha256(root / "target_free_receipts.parquet"),
        "policy_path": str(args.policy_path.resolve()),
        "policy_path_sha256": _sha256(args.policy_path.resolve()),
        "top_base_floor": args.top_base_floor,
        "min_conditional_folds": args.min_conditional_folds,
        "min_conditional_median_bps": args.min_conditional_median_bps,
        "raw_feature_cmi_or_ic": False,
        "outcome_join_happened_after_target_free_receipt_check": True,
        "scope": "offline diagnostic only; no model/admission/portfolio/exchange mutation",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
