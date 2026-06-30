#!/usr/bin/env python3
"""Analyze whether C3el score thresholds rank exact-state utility.

The exact-state labels are intentionally expensive to generate.  This script
combines the existing labeled action panels, joins score metadata where needed,
and writes a compact audit report showing whether p_intervene / predicted
delta-J boundaries map to realized cloned-state utility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEYS = ["timestamp", "strategy_id", "action_family", "action_value"]

DEFAULT_OLD_LABELS = Path(
    "data_perp/reports/c3el_targeted_oracle_shortasset_fallback_states_20260628/"
    "targeted_oracle_vs_replay_delta.csv"
)
DEFAULT_BROAD_LABELS = Path(
    "data_perp/reports/exact_state_counterfactual_oracle_20260628_shortasset_broad_p80_d250_targets/"
    "exact_state_counterfactual_labels.csv"
)
DEFAULT_BROAD_TARGETS = Path(
    "data_perp/reports/c3el_fallback_oracle_targets_20260628_shortasset_broad_p80_d250_v1/"
    "target_actions.csv"
)
DEFAULT_MAY_LABELS = Path(
    "data_perp/reports/exact_state_counterfactual_oracle_20260628_shortasset_may_p70_d100_targets/"
    "exact_state_counterfactual_labels.csv"
)
DEFAULT_MAY_TARGETS = Path(
    "data_perp/reports/c3el_fallback_oracle_targets_20260628_shortasset_may06_may29_p70_d100_v1/"
    "target_actions.csv"
)
DEFAULT_OUT_DIR = Path("data_perp/reports/c3el_exact_score_boundary_audit_20260628")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _normalise_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.loc[out["timestamp"].notna()].copy()
    out["strategy_id"] = out["strategy_id"].astype(str)
    if "action_family" not in out.columns:
        out["action_family"] = "size"
    out["action_family"] = out["action_family"].astype(str)
    if "action_value" not in out.columns:
        if "multiplier" in out.columns:
            out["action_value"] = out["multiplier"]
        else:
            out["action_value"] = np.nan
    out["action_value"] = pd.to_numeric(out["action_value"], errors="coerce").fillna(0.0).round(6)
    return out


def _join_targets(labels: pd.DataFrame, targets: pd.DataFrame, source: str) -> pd.DataFrame:
    lab = _normalise_keys(labels)
    tgt = _normalise_keys(targets)
    score_cols = ["p_intervene", "pred_action_delta_J", "selected_multiplier", "target_priority"]
    present = [c for c in score_cols if c in tgt.columns]
    joined = lab.merge(tgt[KEYS + present], on=KEYS, how="left", validate="one_to_one")
    joined["source"] = source
    return joined


def load_combined(
    *,
    old_labels: Path,
    broad_labels: Path,
    broad_targets: Path,
    may_labels: Path,
    may_targets: Path,
) -> pd.DataFrame:
    old = _normalise_keys(_read_csv(old_labels))
    old["source"] = "original_p80_d320_25"

    broad = _join_targets(_read_csv(broad_labels), _read_csv(broad_targets), "broad_p80_d250_40")
    may = _join_targets(_read_csv(may_labels), _read_csv(may_targets), "may_p70_d100_6")
    combined = pd.concat([old, broad, may], ignore_index=True, sort=False)

    for col in [
        "p_intervene",
        "pred_action_delta_J",
        "delta_full_J",
        "delta_immediate_J",
        "delta_full_net_pnl",
        "delta_full_cost_pnl",
        "delta_full_turnover",
    ]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    if "action_binds" in combined.columns:
        combined["action_binds"] = combined["action_binds"].astype(bool)

    combined = combined.drop_duplicates(KEYS, keep="first").sort_values("timestamp").reset_index(drop=True)
    combined["day"] = combined["timestamp"].dt.floor("D")
    combined["bucket"] = _bucketize(combined)
    return combined


def _bucketize(frame: pd.DataFrame) -> pd.Series:
    p = pd.to_numeric(frame["p_intervene"], errors="coerce")
    delta = pd.to_numeric(frame["pred_action_delta_J"], errors="coerce")
    out = pd.Series("other", index=frame.index, dtype="object")
    out.loc[p.ge(0.80) & delta.ge(320.0)] = "p80_d320"
    out.loc[p.ge(0.80) & delta.ge(250.0) & delta.lt(320.0)] = "p80_d250_320"
    out.loc[p.ge(0.80) & delta.ge(100.0) & delta.lt(250.0)] = "p80_d100_250"
    out.loc[p.ge(0.70) & p.lt(0.80) & delta.ge(100.0)] = "p70_80_d100p"
    return out


def _summary(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="object")
    return pd.Series(
        {
            "rows": int(len(frame)),
            "bind_share": float(frame["action_binds"].mean()),
            "pos_share": float(frame["delta_full_J"].gt(0.0).mean()),
            "pos_gt50_share": float(frame["delta_full_J"].gt(50.0).mean()),
            "sum_delta_full_J": float(frame["delta_full_J"].sum()),
            "mean_delta_full_J": float(frame["delta_full_J"].mean()),
            "median_delta_full_J": float(frame["delta_full_J"].median()),
            "worst_delta_full_J": float(frame["delta_full_J"].min()),
            "sum_immediate_J": float(frame["delta_immediate_J"].sum()),
            "sum_delta_net_pnl": float(frame["delta_full_net_pnl"].sum()),
            "sum_delta_cost_pnl": float(frame["delta_full_cost_pnl"].sum()),
            "sum_delta_turnover": float(frame["delta_full_turnover"].sum()),
            "p_mean": float(frame["p_intervene"].mean()),
            "pred_delta_mean": float(frame["pred_action_delta_J"].mean()),
            "start": frame["timestamp"].min(),
            "end": frame["timestamp"].max(),
        }
    )


def _spearman(frame: pd.DataFrame, feature: str) -> dict[str, Any]:
    work = frame[[feature, "delta_full_J"]].dropna()
    if len(work) < 3 or work[feature].nunique() < 2:
        return {"feature": feature, "rows": int(len(work)), "rho": None}
    return {
        "feature": feature,
        "rows": int(len(work)),
        "rho": float(work[feature].corr(work["delta_full_J"], method="spearman")),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def write_report(combined: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "combined_exact_labels_with_scores.csv", index=False)

    by_source = combined.groupby("source", sort=False).apply(_summary, include_groups=False).reset_index()
    by_bucket = (
        combined.groupby("bucket", sort=False)
        .apply(_summary, include_groups=False)
        .reset_index()
        .sort_values("sum_delta_full_J", ascending=False)
    )
    by_day = combined.groupby("day", sort=True).apply(_summary, include_groups=False).reset_index()
    by_source.to_csv(out_dir / "summary_by_source.csv", index=False)
    by_bucket.to_csv(out_dir / "summary_by_bucket.csv", index=False)
    by_day.to_csv(out_dir / "summary_by_day.csv", index=False)

    spearman = [_spearman(combined, "p_intervene"), _spearman(combined, "pred_action_delta_J")]
    top_negative = combined.sort_values("delta_full_J").head(10)
    top_positive = combined.sort_values("delta_full_J", ascending=False).head(10)

    manifest = {
        "generated_by": "analyze_c3el_exact_score_boundary",
        "rows": int(len(combined)),
        "start": combined["timestamp"].min(),
        "end": combined["timestamp"].max(),
        "sources": combined["source"].value_counts().to_dict(),
        "spearman": spearman,
        "outputs": {
            "combined": str(out_dir / "combined_exact_labels_with_scores.csv"),
            "by_source": str(out_dir / "summary_by_source.csv"),
            "by_bucket": str(out_dir / "summary_by_bucket.csv"),
            "by_day": str(out_dir / "summary_by_day.csv"),
            "summary": str(out_dir / "summary.md"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))

    display_cols = [
        "timestamp",
        "source",
        "bucket",
        "p_intervene",
        "pred_action_delta_J",
        "delta_full_J",
        "delta_immediate_J",
        "action_binds",
    ]
    lines = [
        "# C3el Exact-State Score Boundary Audit",
        "",
        "This report checks whether the current C3el intervention scores rank realized exact-state utility.",
        "",
        f"Rows: `{len(combined)}`",
        f"Period: `{combined['timestamp'].min()}` to `{combined['timestamp'].max()}`",
        "",
        "## By Source",
        "",
        by_source.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## By Score Bucket",
        "",
        by_bucket.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Score Correlation With Exact-State Utility",
        "",
        pd.DataFrame(spearman).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## By Day",
        "",
        by_day.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Worst Exact-State Actions",
        "",
        top_negative[display_cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Best Exact-State Actions",
        "",
        top_positive[display_cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "- Exact-state outcomes are heavily episode dependent. The strict June p80/d320 slice is positive, but broadening around it includes large losers.",
        "- `p_intervene` and `pred_action_delta_J` alone are not sufficient acceptance scores for learned fallback expansion.",
        "- New labels should be collected by forward recurrence of the strict boundary or by a predeclared weak-boundary experiment, with the two tracked separately.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-labels", type=Path, default=DEFAULT_OLD_LABELS)
    parser.add_argument("--broad-labels", type=Path, default=DEFAULT_BROAD_LABELS)
    parser.add_argument("--broad-targets", type=Path, default=DEFAULT_BROAD_TARGETS)
    parser.add_argument("--may-labels", type=Path, default=DEFAULT_MAY_LABELS)
    parser.add_argument("--may-targets", type=Path, default=DEFAULT_MAY_TARGETS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    combined = load_combined(
        old_labels=args.old_labels,
        broad_labels=args.broad_labels,
        broad_targets=args.broad_targets,
        may_labels=args.may_labels,
        may_targets=args.may_targets,
    )
    manifest = write_report(combined, args.out_dir)
    print((args.out_dir / "summary.md").read_text())
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
