#!/usr/bin/env python3
"""Diagnose C3el action-label support under alternative objectives.

Head-native C3el currently fails threshold selection because profitable
interventions are sparse.  This artifact-only diagnostic summarizes whether the
problem is simply an overly strict label, or whether relaxed labels would expose
too much negative-tail action risk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


GROUP_KEYS = ["timestamp", "strategy_id"]
REQUIRED_COLUMNS = {
    "timestamp",
    "strategy_id",
    "group_can_bind",
    "y_intervene",
    "best_gain",
    "best_margin",
    "best_gain_per_notional",
    "best_margin_per_notional",
    "best_immediate_gain",
    "best_nonbaseline_gain",
    "worst_nonbaseline_gain",
}


def _read_frame(path: Path, columns: set[str] | None = None) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=list(columns) if columns else None)
    if columns:
        return pd.read_csv(path, usecols=lambda col: col in columns)
    return pd.read_csv(path)


def _numeric(frame: pd.DataFrame, col: str, *, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def _infer_head(strategy_id: pd.Series) -> pd.Series:
    return strategy_id.astype(str).str.extract(r"^(long_bars|long_dist|short_asset|short_boll)", expand=False).fillna("unknown")


def load_group_panel(paths: list[Path]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in paths:
        frame = _read_frame(path, REQUIRED_COLUMNS)
        missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        parts.append(frame)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.loc[out["timestamp"].notna()].copy()
    out["strategy_id"] = out["strategy_id"].astype(str)
    out = out.drop_duplicates(GROUP_KEYS, keep="last").reset_index(drop=True)
    out["head"] = _infer_head(out["strategy_id"])
    for col in REQUIRED_COLUMNS.difference({"timestamp", "strategy_id"}):
        out[col] = _numeric(out, col)
    return out


def _rate(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else np.nan


def _diagnosis(row: dict[str, Any]) -> str:
    current_rate = float(row.get("current_positive_rate", 0.0) or 0.0)
    relaxed_rate = float(row.get("relaxed_full_positive_rate", 0.0) or 0.0)
    ratio = float(row.get("full_gain_to_worst_abs_ratio", 0.0) or 0.0)
    full_sum = float(row.get("best_nonbaseline_gain_sum", 0.0) or 0.0)
    if full_sum <= 0.0:
        return "negative_oracle_headroom"
    if current_rate < 0.05 and ratio < 0.25:
        return "sparse_low_precision_headroom"
    if current_rate < 0.05 and relaxed_rate > max(current_rate * 2.0, 0.08) and ratio < 0.5:
        return "relaxed_label_tail_risk"
    if current_rate < 0.05:
        return "sparse_but_viable_headroom"
    return "usable_label_support"


def _recommendation(diagnosis: str) -> str:
    if diagnosis == "negative_oracle_headroom":
        return "disable_or_keep_diagnostic; exact-state labels do not show positive size-action headroom"
    if diagnosis == "sparse_low_precision_headroom":
        return "avoid broad relaxed labels; require predeclared guards or regime-conditioned threshold evidence"
    if diagnosis == "relaxed_label_tail_risk":
        return "test relaxed label only with defensive tail guard and leave-one-period validation"
    if diagnosis == "sparse_but_viable_headroom":
        return "test regime-conditioned threshold selection before increasing model capacity"
    return "label_support_is_adequate_for_standard_threshold_selection"


def summarise_objectives(groups: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for head, frame in groups.groupby("head", dropna=False):
        can_bind = _numeric(frame, "group_can_bind").gt(0.0)
        current = _numeric(frame, "y_intervene").gt(0.0)
        relaxed_full = can_bind & _numeric(frame, "best_nonbaseline_gain").gt(0.0)
        relaxed_full_e50 = can_bind & _numeric(frame, "best_nonbaseline_gain").gt(50.0)
        strict_full = (
            can_bind
            & _numeric(frame, "best_gain").gt(50.0)
            & _numeric(frame, "best_margin").gt(25.0)
            & _numeric(frame, "best_gain_per_notional").gt(0.001)
            & _numeric(frame, "best_margin_per_notional").gt(0.0005)
        )
        immediate = can_bind & _numeric(frame, "best_immediate_gain").gt(0.0)
        immediate_e50 = can_bind & _numeric(frame, "best_immediate_gain").gt(50.0)
        best_sum = float(_numeric(frame, "best_nonbaseline_gain").sum())
        worst_sum = float(_numeric(frame, "worst_nonbaseline_gain").sum())
        immediate_sum = float(_numeric(frame, "best_immediate_gain").sum())
        row = {
            "head": str(head),
            "groups": int(len(frame)),
            "can_bind_groups": int(can_bind.sum()),
            "can_bind_rate": _rate(can_bind),
            "current_positive_count": int(current.sum()),
            "current_positive_rate": _rate(current),
            "relaxed_full_positive_count": int(relaxed_full.sum()),
            "relaxed_full_positive_rate": _rate(relaxed_full),
            "relaxed_full_e50_count": int(relaxed_full_e50.sum()),
            "relaxed_full_e50_rate": _rate(relaxed_full_e50),
            "strict_full_positive_count": int(strict_full.sum()),
            "strict_full_positive_rate": _rate(strict_full),
            "immediate_positive_count": int(immediate.sum()),
            "immediate_positive_rate": _rate(immediate),
            "immediate_e50_count": int(immediate_e50.sum()),
            "immediate_e50_rate": _rate(immediate_e50),
            "best_nonbaseline_gain_sum": best_sum,
            "best_nonbaseline_gain_median": float(_numeric(frame, "best_nonbaseline_gain").median()),
            "best_nonbaseline_gain_q95": float(_numeric(frame, "best_nonbaseline_gain").quantile(0.95)),
            "best_immediate_gain_sum": immediate_sum,
            "best_immediate_gain_median": float(_numeric(frame, "best_immediate_gain").median()),
            "best_immediate_gain_q95": float(_numeric(frame, "best_immediate_gain").quantile(0.95)),
            "worst_nonbaseline_gain_sum": worst_sum,
            "worst_nonbaseline_gain_q05": float(_numeric(frame, "worst_nonbaseline_gain").quantile(0.05)),
            "full_gain_to_worst_abs_ratio": float(best_sum / max(abs(worst_sum), 1e-9)),
            "immediate_to_full_gain_ratio": float(immediate_sum / max(abs(best_sum), 1e-9)),
        }
        row["diagnosis"] = _diagnosis(row)
        row["recommendation"] = _recommendation(str(row["diagnosis"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["diagnosis", "head"]).reset_index(drop=True)


def _write_markdown(path: Path, report: pd.DataFrame) -> None:
    lines = [
        "# C3el action-label objective diagnostics",
        "",
        "This report compares current and relaxed exact-state size-action labels by head.",
        "",
    ]
    if report.empty:
        lines.append("No rows.")
    else:
        cols = [
            "head",
            "diagnosis",
            "recommendation",
            "groups",
            "can_bind_rate",
            "current_positive_rate",
            "relaxed_full_positive_rate",
            "relaxed_full_e50_rate",
            "immediate_positive_rate",
            "best_nonbaseline_gain_sum",
            "worst_nonbaseline_gain_sum",
            "full_gain_to_worst_abs_ratio",
            "best_nonbaseline_gain_q95",
            "worst_nonbaseline_gain_q05",
        ]
        lines.append(report[cols].to_markdown(index=False, floatfmt=".4f"))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `sparse_low_precision_headroom`: current positives are rare and positive oracle headroom is small versus the negative tail.",
            "- `relaxed_label_tail_risk`: a relaxed positive label creates more positives, but likely admits too many damaging actions.",
            "- `sparse_but_viable_headroom`: positives are rare, but positive headroom is large enough to justify regime-conditioned threshold tests.",
            "",
            "## Next ablation hypothesis",
            "",
            "Do not simply lower the positive threshold when the full-gain-to-worst ratio is poor. Use predeclared guards, regime-conditioned thresholds, or collect more forward exact-state labels for the sparse slices that already showed defensive success.",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    groups = load_group_panel(list(args.panel))
    report = summarise_objectives(groups)
    report.to_csv(args.out_dir / "action_label_objective_diagnostics.csv", index=False)
    _write_markdown(args.out_dir / "summary.md", report)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "generated_by": "diagnose_c3el_action_label_objectives",
                "panels": [str(p) for p in args.panel],
                "group_rows": int(len(groups)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
