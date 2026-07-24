#!/usr/bin/env python3
"""Compare meta rankers strictly inside the short candidate stream.

This is intentionally not a portfolio or global-auction report.  Each model is
ranked only against the same short OOS candidates, which isolates short ranking
quality from long-score dominance and cross-side score comparability.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["__ts__", "__symbol__", "side_name"]
OUTCOME_COLUMNS = [
    "ev_after_1pct",
    "first_touch_gross",
    "clean_exec",
    "dirty_positive",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "timeout",
    "archetype_label_family",
    "month",
]
SCORE_COLUMN = "score_meta_base_soft_label"
TAILS = (0.01, 0.05, 0.10, 0.20, 0.30)


def _read_short(path: Path) -> pd.DataFrame:
    requested = list(dict.fromkeys([*KEYS, SCORE_COLUMN, *OUTCOME_COLUMNS]))
    frame = pd.read_parquet(path, columns=requested)
    frame = frame.loc[frame["side_name"].eq("short")].copy()
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    frame["month"] = frame["__ts__"].dt.to_period("M").astype(str)
    frame["week_start"] = (
        frame["__ts__"].dt.normalize()
        - pd.to_timedelta(frame["__ts__"].dt.dayofweek, unit="D")
    )
    for column in [SCORE_COLUMN, *OUTCOME_COLUMNS[:-2]]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _top(frame: pd.DataFrame, tail: float) -> pd.DataFrame:
    n = max(1, int(np.ceil(len(frame) * tail)))
    return frame.nlargest(n, SCORE_COLUMN, keep="first")


def _metrics(frame: pd.DataFrame, model: str, tail: float, scope: str, value: str) -> dict[str, object]:
    selected = _top(frame, tail)
    ev = pd.to_numeric(selected["ev_after_1pct"], errors="coerce")
    gross = pd.to_numeric(selected["first_touch_gross"], errors="coerce")
    row: dict[str, object] = {
        "model": model,
        "tail": tail,
        "scope": scope,
        "scope_value": value,
        "candidate_rows": len(frame),
        "selected_rows": len(selected),
        "mean_net_ev_after_1pct": ev.mean(),
        "sum_net_ev_after_1pct": ev.sum(),
        "mean_gross_ev": gross.mean(),
        "positive_ev_rate": ev.gt(0.0).mean(),
        "clean_exec_rate": pd.to_numeric(selected["clean_exec"], errors="coerce").mean(),
        "dirty_positive_rate": pd.to_numeric(selected["dirty_positive"], errors="coerce").mean(),
        "first_touch_bad_mae_rate": pd.to_numeric(selected["first_touch_bad_mae_1r"], errors="coerce").mean(),
        "full_path_bad_mae_rate": pd.to_numeric(selected["full_path_bad_mae_1r"], errors="coerce").mean(),
        "timeout_rate": pd.to_numeric(selected["timeout"], errors="coerce").mean(),
        "mean_score": pd.to_numeric(selected[SCORE_COLUMN], errors="coerce").mean(),
    }
    return row


def _scope_metrics(frame: pd.DataFrame, model: str, group: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for value, part in frame.groupby(group, sort=True, observed=True):
        for tail in TAILS:
            rows.append(_metrics(part, model, tail, group, str(value)))
    return rows


def _calibration(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    work = frame.copy()
    work["score_decile"] = pd.qcut(
        work[SCORE_COLUMN].rank(method="first"), q=10, labels=False, duplicates="drop"
    )
    return (
        work.groupby("score_decile", observed=True)
        .agg(
            rows=(SCORE_COLUMN, "size"),
            mean_score=(SCORE_COLUMN, "mean"),
            mean_net_ev_after_1pct=("ev_after_1pct", "mean"),
            mean_gross_ev=("first_touch_gross", "mean"),
            clean_exec_rate=("clean_exec", "mean"),
            dirty_positive_rate=("dirty_positive", "mean"),
            first_touch_bad_mae_rate=("first_touch_bad_mae_1r", "mean"),
        )
        .reset_index()
        .assign(model=model)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weighted", type=Path, required=True)
    parser.add_argument("--purged", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = {
        "weighted_pack_b": _read_short(args.weighted),
        "current_purged_meta": _read_short(args.purged),
    }
    left = models["weighted_pack_b"]
    right = models["current_purged_meta"]
    overlap = left.merge(right.loc[:, [*KEYS, SCORE_COLUMN]], on=KEYS, how="inner", suffixes=("_weighted", "_purged"))
    # The two historical artifacts can legitimately have different candidate
    # universes.  Keep their within-short metrics separate and expose that
    # fact instead of silently treating them as an exact comparison.
    comparable_universe = len(overlap) == len(left) == len(right)

    metrics: list[dict[str, object]] = []
    calibrations: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    for name, frame in models.items():
        for tail in TAILS:
            metrics.append(_metrics(frame, name, tail, "overall", "all"))
        metrics.extend(_scope_metrics(frame, name, "month"))
        metrics.extend(_scope_metrics(frame, name, "week_start"))
        metrics.extend(_scope_metrics(frame, name, "archetype_label_family"))
        calibrations.append(_calibration(frame, name))
        for tail in TAILS:
            selected = _top(frame, tail).loc[:, KEYS]
            selected["model"] = name
            selected["tail"] = tail
            selection_rows.append({"model": name, "tail": tail, "selected_rows": len(selected)})
            selected.to_parquet(args.output_dir / f"{name}_short_top{int(tail * 100):02d}_keys.parquet", index=False)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(args.output_dir / "short_only_metrics.csv", index=False)
    pd.concat(calibrations, ignore_index=True).to_csv(args.output_dir / "short_score_decile_calibration.csv", index=False)

    overlap_rows: list[dict[str, object]] = []
    for tail in TAILS:
        a = _top(left, tail).loc[:, KEYS]
        b = _top(right, tail).loc[:, KEYS]
        shared = a.merge(b, on=KEYS, how="inner")
        overlap_rows.append(
            {
                "tail": tail,
                "weighted_selected": len(a),
                "purged_selected": len(b),
                "intersection": len(shared),
                "jaccard": len(shared) / (len(a) + len(b) - len(shared)),
            }
        )
    pd.DataFrame(overlap_rows).to_csv(args.output_dir / "short_selection_overlap.csv", index=False)

    # A strict same-row comparison is smaller when two historical candidate
    # universes differ, but it is the only valid head-to-head ranking result.
    common_keys = overlap.loc[:, KEYS]
    common_metrics: list[dict[str, object]] = []
    outcome_max_abs_deltas: dict[str, float] = {}
    for outcome in ["ev_after_1pct", "first_touch_gross", "clean_exec", "dirty_positive", "first_touch_bad_mae_1r", "timeout"]:
        paired = left.loc[:, [*KEYS, outcome]].merge(
            right.loc[:, [*KEYS, outcome]], on=KEYS, how="inner", suffixes=("_weighted", "_purged")
        )
        delta = pd.to_numeric(paired[f"{outcome}_weighted"], errors="coerce") - pd.to_numeric(
            paired[f"{outcome}_purged"], errors="coerce"
        )
        outcome_max_abs_deltas[outcome] = float(delta.abs().max())
    for name, frame in models.items():
        common = frame.merge(common_keys, on=KEYS, how="inner")
        for tail in TAILS:
            common_metrics.append(_metrics(common, name, tail, "exact_common_overall", "all"))
    pd.DataFrame(common_metrics).to_csv(args.output_dir / "short_exact_common_metrics.csv", index=False)

    summary = {
        "scope": "short-only OOS candidates; no long rows were ranked or selected",
        "cost_contract": "ev_after_1pct already contains exactly one fixed 1% round-trip cost",
        "weighted_input": str(args.weighted),
        "purged_input": str(args.purged),
        "short_rows": int(len(left)),
        "short_key_overlap": int(len(overlap)),
        "exact_common_universe": comparable_universe,
        "score_spearman_on_overlap": float(overlap[f"{SCORE_COLUMN}_weighted"].corr(overlap[f"{SCORE_COLUMN}_purged"], method="spearman")),
        "outcome_max_abs_deltas_on_overlap": outcome_max_abs_deltas,
    }
    pd.Series(summary).to_json(args.output_dir / "manifest.json", indent=2)


if __name__ == "__main__":
    main()
